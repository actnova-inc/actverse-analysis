"""
Colab 셀 실행 텔레메트리 (ACT-5939).

노트북이 ``import actverse`` 를 실행하는 순간 IPython 의 ``pre_run_cell`` /
``post_run_cell`` 이벤트에 훅을 등록한다. 이후 사용자가 어떤 셀을 실행하든
(원본 셀이든, 고쳐 쓴 셀이든, 새로 만든 셀이든) 실행된 셀 원문·실행 순번·
성공/실패·traceback·stdout 이 한 건의 이벤트로 묶여 actverse-api 로 전송된다.

설계 원칙
- 사용자 셀 실행에 절대 영향을 주지 않는다: 모든 훅 본문은 예외를 삼키고,
  전송은 데몬 스레드에서 짧은 timeout 으로 한다. 수신 API 가 죽어 있어도
  노트북은 평소처럼 돈다.
- 데이터 본문(예측 JSON·좌표·그래프)은 보내지 않는다. 셀 원문·traceback·
  stdout 은 상한을 두고 잘라 보낸다.
- 사용자 신원(Google 계정·IP)은 수집하지 않는다. 식별은 노트북에 심어진
  ``analysis_id`` / ``video_id`` 로만 한다.

식별자 주입 (우선순위)
1. ``actverse.telemetry.configure(analysis_id=..., video_id=...)`` 명시 호출
2. 노트북 커널 네임스페이스의 ``ACTVERSE_ANALYSIS_ID`` / ``ACTVERSE_VIDEO_ID``
   변수 (템플릿 2번 셀에 자리표시로 들어가고 actverse-api 가 치환한다)
3. 환경변수 ``ACTVERSE_ANALYSIS_ID`` / ``ACTVERSE_VIDEO_ID``
셀마다 위 1~3 을 다시 확인한다 — 사용자가 import 뒤에 id 를 적어도 잡힌다.

비활성화: 환경변수 ``ACTVERSE_TELEMETRY=0`` 또는 ``disable()`` 호출.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import threading
import time
import traceback as _tb
from typing import Any, Optional

__all__ = [
    "configure",
    "enable",
    "disable",
    "is_enabled",
    "record_call",
    "DEFAULT_ENDPOINT",
]

# ACT-5940 수신 API 계약: POST {endpoint}/analyses/{analysis_id}/colab-events
DEFAULT_ENDPOINT = os.environ.get(
    "ACTVERSE_TELEMETRY_ENDPOINT", "https://ml.actverse.io/api/v2"
)

# 상한 — 수신 API 페이로드 상한(16KB) 안에 들어오도록 각 필드를 자른다.
MAX_CELL_SOURCE = 6 * 1024
MAX_TRACEBACK = 4 * 1024
MAX_STDOUT = 2 * 1024
MAX_CALLS_PER_CELL = 20
SEND_TIMEOUT_S = 3.0

# 템플릿 원본 셀 해시 → 그 셀의 줄 지문 목록. bin/_templates/build.py 가
# actverse/_template_hashes.py 로 생성한다. ko/en 둘의 해시를 모두 담는다.
# 해시 일치 → ``original``, 어느 템플릿 셀과 줄 지문이 충분히 겹치면 → ``modified``,
# 아니면 ``new``. 목록이 비어 있으면 ``unknown``.
TEMPLATE_CELL_HASHES: dict[str, list] = {}
try:
    from ._template_hashes import TEMPLATE_CELL_HASHES as _gen  # type: ignore

    TEMPLATE_CELL_HASHES.update(_gen)
except Exception:  # pragma: no cover
    pass

# 사용자가 고친 셀로 인정할 최소 줄 겹침 비율 (Jaccard). 0.5 = 절반 이상이 원본 줄.
MODIFIED_MIN_SIMILARITY = 0.5


class _State:
    enabled: bool = os.environ.get("ACTVERSE_TELEMETRY", "1") not in ("0", "false", "no")
    registered: bool = False
    shell: Any = None
    endpoint: str = DEFAULT_ENDPOINT
    analysis_id: Optional[str] = None
    video_id: Optional[str] = None
    session_started_at: float = time.time()
    session_id: str = hashlib.sha1(f"{time.time()}-{os.getpid()}".encode()).hexdigest()[:12]
    current: Optional[dict] = None
    current_calls: list = []
    _stdout_capture: Any = None
    _orig_stdout: Any = None


_state = _State()


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def configure(
    analysis_id: Optional[str] = None,
    video_id: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> None:
    """식별자·엔드포인트를 명시 설정한다. 노트북 셀에서 직접 호출해도 된다."""
    if analysis_id:
        _state.analysis_id = str(analysis_id)
    if video_id:
        _state.video_id = str(video_id)
    if endpoint:
        _state.endpoint = endpoint.rstrip("/")


def enable() -> bool:
    """IPython 이벤트에 훅을 등록한다. IPython 밖(일반 파이썬)에서는 no-op."""
    _state.enabled = True
    return _register()


def disable() -> None:
    """훅을 해제한다. 이후 셀 실행은 전송되지 않는다."""
    _state.enabled = False
    _unregister()


def is_enabled() -> bool:
    return _state.enabled and _state.registered


def record_call(name: str, **summary: Any) -> None:
    """actverse 패키지 내부 함수가 자기 호출을 요약해 남긴다.

    값 본문이 아니라 크기·개수·이름 같은 요약만 넘길 것. 현재 실행 중인 셀
    이벤트의 ``calls`` 배열에 붙어 함께 전송된다. 셀 밖(훅 미등록)에서 호출되면
    조용히 버린다.
    """
    try:
        if not _state.registered or _state.current is None:
            return
        if len(_state.current_calls) >= MAX_CALLS_PER_CELL:
            return
        _state.current_calls.append({"fn": name, **_safe_json(summary)})
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# IPython hooks
# --------------------------------------------------------------------------- #
def _get_ipython():
    try:
        from IPython import get_ipython  # type: ignore

        return get_ipython()
    except Exception:
        return None


def _register() -> bool:
    if _state.registered:
        return True
    if not _state.enabled:
        return False
    ip = _get_ipython()
    if ip is None:
        return False
    try:
        ip.events.register("pre_run_cell", _pre_run_cell)
        ip.events.register("post_run_cell", _post_run_cell)
        _state.shell = ip
        _state.registered = True
        _adopt_in_flight_cell(ip)
        return True
    except Exception:
        return False


def _adopt_in_flight_cell(ip) -> None:
    """등록이 셀 실행 도중(``import actverse`` 셀)에 일어나면 pre_run_cell 은
    이미 지나갔다. 그 셀도 기록되도록 현재 입력을 컨텍스트로 삼는다.
    템플릿 2번 셀(import + id + load_json)이 바로 이 경우다."""
    try:
        if _state.current is not None:
            return
        raw = ""
        # ① ipykernel(Colab/Jupyter): 실행 중인 execute_request 의 code
        try:
            parent = ip.get_parent() if hasattr(ip, "get_parent") else None
            raw = ((parent or {}).get("content") or {}).get("code") or ""
        except Exception:
            raw = ""
        # ② fallback: run_cell(store_history=True) 는 실행 전에 In 에 넣는다
        if not raw.strip():
            inputs = (getattr(ip, "user_ns", None) or {}).get("In") or []
            raw = (inputs[-1] if inputs else "") or ""
        if not raw.strip():
            return
        _state.current = {
            "started_at": time.time(),
            "cell_source_full": raw,
            "cell_id": None,
            "adopted_mid_cell": True,
        }
        _state.current_calls = []
    except Exception:
        _state.current = None


def _unregister() -> None:
    ip = _state.shell or _get_ipython()
    if ip is None:
        _state.registered = False
        return
    for name, fn in (("pre_run_cell", _pre_run_cell), ("post_run_cell", _post_run_cell)):
        try:
            ip.events.unregister(name, fn)
        except Exception:
            pass
    _state.registered = False


class _Tee:
    """stdout 을 화면에 그대로 흘리면서 앞부분만 복사해 둔다."""

    def __init__(self, orig, limit: int):
        self._orig = orig
        self._buf: list = []
        self._size = 0
        self._limit = limit
        self.truncated = False

    def write(self, s):
        try:
            if self._size < self._limit:
                take = s[: self._limit - self._size]
                self._buf.append(take)
                self._size += len(take)
                if len(take) < len(s):
                    self.truncated = True
            else:
                self.truncated = True
        except Exception:
            pass
        return self._orig.write(s)

    def flush(self):
        return self._orig.flush()

    def __getattr__(self, item):  # isatty, encoding 등은 원본에 위임
        return getattr(self._orig, item)

    def text(self) -> str:
        return "".join(self._buf)


def _pre_run_cell(info) -> None:
    try:
        if not _state.enabled:
            return
        raw = getattr(info, "raw_cell", None) or ""
        _state.current = {
            "started_at": time.time(),
            "cell_source_full": raw,
            "cell_id": getattr(info, "cell_id", None),
        }
        _state.current_calls = []
        _state._orig_stdout = sys.stdout
        _state._stdout_capture = _Tee(sys.stdout, MAX_STDOUT)
        sys.stdout = _state._stdout_capture
    except Exception:
        _state.current = None


def _post_run_cell(result) -> None:
    try:
        # stdout 복원은 어떤 경우에도 먼저 한다
        cap = _state._stdout_capture
        if _state._orig_stdout is not None:
            sys.stdout = _state._orig_stdout
        _state._orig_stdout = None
        _state._stdout_capture = None

        if not _state.enabled:
            return
        cur = _state.current
        _state.current = None
        if cur is None:
            return

        # 식별자는 셀마다 다시 확인 — 사용자가 import 뒤에 적어도 잡힌다
        _refresh_ids_from_namespace()

        raw = cur.get("cell_source_full") or getattr(getattr(result, "info", None), "raw_cell", "") or ""
        cell_hash = _sha256(raw)
        error_before = getattr(result, "error_before_exec", None)
        error_in = getattr(result, "error_in_exec", None)
        err = error_before or error_in

        event = {
            "schema": "actverse_colab_event_v1",
            "session_id": _state.session_id,
            "analysis_id": _state.analysis_id or "unknown",
            "video_id": _state.video_id or "unknown",
            "execution_count": getattr(result, "execution_count", None),
            "started_at": _iso(cur["started_at"]),
            "duration_ms": int((time.time() - cur["started_at"]) * 1000),
            "cell_id": cur.get("cell_id"),
            "adopted_mid_cell": bool(cur.get("adopted_mid_cell", False)),
            "cell_source": _truncate(raw, MAX_CELL_SOURCE),
            "cell_source_truncated": len(raw) > MAX_CELL_SOURCE,
            "cell_hash": cell_hash,
            "cell_kind": _classify(raw, cell_hash),
            "status": "error" if err is not None else "ok",
            "error_type": type(err).__name__ if err is not None else None,
            "error_message": _truncate(str(err), 1024) if err is not None else None,
            "error_phase": (
                "before_exec" if error_before is not None
                else ("in_exec" if error_in is not None else None)
            ),
            "traceback": _format_tb(err) if err is not None else None,
            "stdout": _clean_stdout(cap.text()) if cap is not None else "",
            "stdout_truncated": bool(cap.truncated) if cap is not None else False,
            "calls": list(_state.current_calls),
            "env": _env_snapshot(),
        }
        _state.current_calls = []
        _send_async(event)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _refresh_ids_from_namespace() -> None:
    try:
        ns = getattr(_state.shell, "user_ns", None) or {}
        a = ns.get("ACTVERSE_ANALYSIS_ID") or os.environ.get("ACTVERSE_ANALYSIS_ID")
        v = ns.get("ACTVERSE_VIDEO_ID") or os.environ.get("ACTVERSE_VIDEO_ID")
        if a and not str(a).startswith("__"):  # 치환 안 된 자리표시는 무시
            _state.analysis_id = str(a)
        if v and not str(v).startswith("__"):
            _state.video_id = str(v)
    except Exception:
        pass


def _classify(raw: str, cell_hash: str) -> str:
    if not TEMPLATE_CELL_HASHES:
        return "unknown"
    if cell_hash in TEMPLATE_CELL_HASHES:
        return "original"
    mine = set(line_fingerprints(raw))
    if not mine:
        return "new"
    best = 0.0
    for fps in TEMPLATE_CELL_HASHES.values():
        theirs = set(fps or [])
        if not theirs:
            continue
        inter = len(mine & theirs)
        if inter == 0:
            continue
        best = max(best, inter / len(mine | theirs))
    return "modified" if best >= MODIFIED_MIN_SIMILARITY else "new"


def line_fingerprints(raw: str) -> list:
    """셀의 의미 있는 줄(빈 줄·주석 제외)마다 짧은 지문. build.py 와 공유.

    ``normalize_source`` 를 거치므로 서버가 치환하는 줄(id·json_path)은 값과
    무관하게 같은 지문이 된다."""
    out = []
    for line in normalize_source(raw).split("\n"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(hashlib.sha1(s.encode("utf-8")).hexdigest()[:12])
    return out


_SERVER_SUBSTITUTED_PREFIXES = ("ACTVERSE_ANALYSIS_ID", "ACTVERSE_VIDEO_ID", "json_path")


def normalize_source(raw: str) -> str:
    """분류용 정규화 (build.py 와 공유).

    - 줄 끝 공백·CRLF·마지막 개행 차이는 같은 셀로 본다.
    - actverse-api 가 노트북 발행 시 치환하는 줄(``ACTVERSE_*_ID = ...``,
      ``json_path = ...``)은 값이 사용자마다 다르므로 좌변만 남긴다. 그래야
      서버가 채운 2번 셀이 ``modified`` 가 아니라 ``original`` 로 분류된다.
    """
    out = []
    for line in raw.replace("\r\n", "\n").split("\n"):
        line = line.rstrip()
        stripped = line.lstrip()
        for prefix in _SERVER_SUBSTITUTED_PREFIXES:
            if stripped.startswith(prefix) and "=" in stripped:
                line = line[: len(line) - len(stripped)] + prefix + " = <substituted>"
                break
        out.append(line)
    return "\n".join(out).strip()


def _sha256(raw: str) -> str:
    return hashlib.sha256(normalize_source(raw).encode("utf-8")).hexdigest()


_ANSI_RE = None


def _clean_stdout(text: str) -> str:
    """ANSI 색 코드 제거. IPython 이 stdout 에 다시 찍는 traceback 은 별도
    ``traceback`` 필드와 중복이므로 그 구분선 이후는 잘라낸다."""
    global _ANSI_RE
    try:
        import re

        if _ANSI_RE is None:
            _ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
        text = _ANSI_RE.sub("", text)
        marker = "\n" + "-" * 75
        idx = text.find(marker)
        if idx == -1 and text.startswith("-" * 75):
            idx = 0
        if idx != -1:
            text = text[:idx].rstrip("\n")
    except Exception:
        pass
    return text


def _truncate(s: Optional[str], n: int) -> str:
    if s is None:
        return ""
    return s if len(s) <= n else s[:n]


def _format_tb(err) -> str:
    try:
        text = "".join(_tb.format_exception(type(err), err, err.__traceback__))
    except Exception:
        text = repr(err)
    if len(text) > MAX_TRACEBACK:
        # 앞은 원인, 뒤는 마지막 프레임이 중요하므로 양끝을 남긴다
        half = MAX_TRACEBACK // 2
        text = text[:half] + "\n...[truncated]...\n" + text[-half:]
    return text


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)) + f".{int((ts % 1) * 1000):03d}Z"


_env_cache: Optional[dict] = None


def _env_snapshot() -> dict:
    global _env_cache
    if _env_cache is None:
        env: dict = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "is_colab": _is_colab(),
            "actverse_version": _pkg_version("actverse"),
            "actverse_commit": _pkg_commit(),
        }
        for mod in ("IPython", "plotly", "numpy", "ipywidgets"):
            env[mod.lower()] = _pkg_version(mod)
        _env_cache = env
    out = dict(_env_cache)
    out["runtime_uptime_s"] = int(time.time() - _state.session_started_at)
    return out


def _is_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:
        return "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ


def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        try:
            mod = __import__(name)
            return getattr(mod, "__version__", None)
        except Exception:
            return None


def _pkg_commit() -> Optional[str]:
    """pip install git+... 로 설치되면 direct_url.json 에 커밋이 남는다."""
    try:
        from importlib.metadata import distribution

        raw = distribution("actverse").read_text("direct_url.json")
        if raw:
            return (json.loads(raw).get("vcs_info") or {}).get("commit_id")
    except Exception:
        pass
    return None


def _safe_json(obj: Any) -> Any:
    """JSON 직렬화 가능한 형태로 얕게 정리한다 (numpy 등 대비)."""
    try:
        json.dumps(obj)
        return obj
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_safe_json(v) for v in list(obj)[:50]]
    try:
        return obj.item()  # numpy scalar
    except Exception:
        return str(obj)[:200]


def _send_async(event: dict) -> None:
    threading.Thread(
        target=_send, args=(event,), daemon=True, name="actverse-telemetry"
    ).start()


def _send(event: dict) -> None:
    try:
        import requests

        url = f"{_state.endpoint}/analyses/{event['analysis_id']}/colab-events"
        requests.post(
            url,
            json=event,
            timeout=SEND_TIMEOUT_S,
            headers={"User-Agent": f"actverse-telemetry/{_pkg_version('actverse') or '0'}"},
        )
    except Exception:
        # 전송 실패는 조용히 버린다 — 사용자 화면에 아무것도 남기지 않는다
        pass


# import 시 자동 등록 (IPython 밖이면 no-op)
_register()
