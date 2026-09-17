"""ACT-5939 텔레메트리 end-to-end 테스트.

실제 IPython InteractiveShell 에서 셀을 실행하고, 로컬 HTTP 수신기로 들어온
이벤트를 검증한다. 실행: python tests/test_telemetry.py  (pytest 도 가능)
"""
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RECEIVED: list = []


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        RECEIVED.append({"path": self.path, "body": json.loads(body), "size": n})
        self.send_response(202)
        self.end_headers()

    def log_message(self, *a):
        pass


def _start_server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _wait(n=None, timeout=5.0):
    """n 이 None 이면 '이전 호출 이후 새 이벤트 1건' 을 기다린다."""
    global _SEEN
    target = (_SEEN + 1) if n is None else n
    t0 = time.time()
    while len(RECEIVED) < target and time.time() - t0 < timeout:
        time.sleep(0.05)
    ok = len(RECEIVED) >= target
    _SEEN = len(RECEIVED)
    return ok


_SEEN = 0


def _fresh_shell():
    from IPython.core.interactiveshell import InteractiveShell

    InteractiveShell.clear_instance()
    return InteractiveShell.instance()


def _purge_actverse():
    for m in [k for k in sys.modules if k.startswith("actverse")]:
        del sys.modules[m]


def run_all():
    import contextlib, io
    # IPython 이 stdout 에 그대로 찍는 셀 출력·traceback 은 테스트 로그를 어지럽히므로
    # 실제 stdout 을 잠시 바꿔 둔다 (텔레메트리 Tee 는 그 위에 얹힌다).
    real_stdout = sys.stdout
    quiet = io.StringIO()

    def say(msg):
        real_stdout.write(msg + "\n"); real_stdout.flush()

    sys.stdout = quiet
    try:
        _run_all(say)
    finally:
        sys.stdout = real_stdout


def _run_all(print):
    srv = _start_server()
    os.environ["ACTVERSE_TELEMETRY_ENDPOINT"] = f"http://127.0.0.1:{srv.server_address[1]}/api/v2"
    _purge_actverse()
    ip = _fresh_shell()

    nb = json.loads((REPO / "bin/ko/custom_analysis.ipynb").read_text(encoding="utf-8"))
    code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

    # 1. 서버가 id·json_path 를 치환한 2번 셀 → original
    cell2 = (
        code_cells[1]
        .replace("__ANALYSIS_ID__", "an_test_123")
        .replace("__VIDEO_ID__", "vid_test_456")
        .replace('input("Downloadable url or local file path:")', '"https://drive.usercontent.google.com/download?id=abc"')
        .replace("prediction = load_json(json_path)", "prediction = None  # skip load in test")
    )
    r = ip.run_cell(store_history=True, raw_cell=cell2)
    assert r.success, r.error_in_exec
    assert _wait(), "no event for cell2"
    ev = RECEIVED[-1]["body"]
    assert RECEIVED[-1]["path"] == "/api/v2/analyses/an_test_123/colab-events", RECEIVED[-1]["path"]
    assert ev["analysis_id"] == "an_test_123" and ev["video_id"] == "vid_test_456"
    assert ev["status"] == "ok" and ev["execution_count"] == 1
    # 마지막 줄을 고쳤으므로 modified 가 맞다; 원문 그대로면 original 이어야 한다
    assert ev["cell_kind"] == "modified", ev["cell_kind"]
    print("1  filled cell2 (edited last line)  OK  kind=%s exec=%s path=%s" % (ev["cell_kind"], ev["execution_count"], RECEIVED[-1]["path"]))

    # 1b. 원문 그대로 (id·json_path 만 치환) → original
    cell2_pure = (
        code_cells[1]
        .replace("__ANALYSIS_ID__", "an_test_123")
        .replace("__VIDEO_ID__", "vid_test_456")
        .replace('input("Downloadable url or local file path:")', '"/tmp/actverse_test_pred.json"')
    )
    Path("/tmp/actverse_test_pred.json").write_text(json.dumps({"metadata": {"origin_width": 640, "origin_height": 480}, "results": []}))
    r = ip.run_cell(store_history=True, raw_cell=cell2_pure)
    assert r.success, r.error_in_exec
    assert _wait()
    ev = RECEIVED[-1]["body"]
    assert ev["cell_kind"] == "original", ev["cell_kind"]
    assert any(c["fn"] == "load_json" for c in ev["calls"]), ev["calls"]
    print("1b pure cell2 (server-substituted)  OK  kind=%s calls=%s" % (ev["cell_kind"], [c["fn"] for c in ev["calls"]]))

    # 2. 원본 셀을 고쳐 실행 → modified + 수정 원문 수신
    modified = code_cells[3].replace("body_parts = get_checked(checkboxes)", "body_parts = []  # user edit")
    ip.run_cell(store_history=True, raw_cell=modified)
    assert _wait()
    ev = RECEIVED[-1]["body"]
    assert ev["cell_kind"] == "modified", ev["cell_kind"]
    assert "# user edit" in ev["cell_source"]
    print("2  modified metrics cell            OK  kind=%s status=%s error=%s" % (ev["cell_kind"], ev["status"], ev["error_type"]))

    # 2b. 템플릿 셀과 첫 줄(import numpy as np)만 같은 새 셀 → new (첫 줄 휴리스틱 회귀 방지)
    ip.run_cell(store_history=True, raw_cell="import numpy as np\nfoo = np.zeros(3)\nfoo.sum()")
    assert _wait()
    ev = RECEIVED[-1]["body"]
    assert ev["cell_kind"] == "new", ev["cell_kind"]
    print("2b new cell sharing an import line   OK  kind=%s" % ev["cell_kind"])

    # 3. 새 셀 + NameError → new / error / traceback / stdout
    ip.run_cell(store_history=True, raw_cell="x = 1\nprint('hello stdout')\nall_metrics[0]")
    assert _wait()
    ev = RECEIVED[-1]["body"]
    assert ev["cell_kind"] == "new"
    assert ev["status"] == "error" and ev["error_type"] == "NameError", (ev["status"], ev["error_type"])
    assert "NameError" in ev["traceback"] and "hello stdout" in ev["stdout"]
    assert "Traceback" not in ev["stdout"] and "\x1b[" not in ev["stdout"], ev["stdout"]
    assert ev["error_phase"] == "in_exec"
    print("3  new cell w/ NameError            OK  tb=%d chars stdout=%r" % (len(ev["traceback"]), ev["stdout"].strip()))

    # 4. 문법 오류 → before_exec
    ip.run_cell(store_history=True, raw_cell="def broken(:\n  pass")
    assert _wait()
    ev = RECEIVED[-1]["body"]
    assert ev["status"] == "error" and ev["error_phase"] == "before_exec", ev
    print("4  syntax error                     OK  type=%s phase=%s" % (ev["error_type"], ev["error_phase"]))

    # 5. 페이로드 상한
    ip.run_cell(store_history=True, raw_cell="print('A'*50000)\nraise RuntimeError('B'*50000)")
    assert _wait()
    size = RECEIVED[-1]["size"]
    assert size < 16 * 1024, f"payload {size} bytes exceeds 16KB"
    assert RECEIVED[-1]["body"]["stdout_truncated"] is True
    print("5  payload cap                      OK  size=%d bytes" % size)

    # 6. 수신 API 다운 → 노트북 영향 없음
    srv.shutdown()
    t0 = time.time()
    r = ip.run_cell(store_history=True, raw_cell="y = 2 + 2\ny")
    dt = time.time() - t0
    assert r.success and r.result == 4 and dt < 1.0, (r.success, dt)
    print("6  API down -> no impact            OK  cell took %.3fs" % dt)

    # 7. 세션 재시작 → 재등록, execution_count 1 부터
    srv2 = _start_server()
    os.environ["ACTVERSE_TELEMETRY_ENDPOINT"] = f"http://127.0.0.1:{srv2.server_address[1]}/api/v2"
    _purge_actverse()
    before = len(RECEIVED)
    ip2 = _fresh_shell()
    ip2.run_cell(store_history=True, raw_cell="import actverse\nz = 1")
    assert _wait(before + 1)
    ev = RECEIVED[-1]["body"]
    assert ev["execution_count"] == 1 and ev["analysis_id"] == "unknown", ev
    print("7  restart -> re-register           OK  exec=%s analysis_id=%s" % (ev["execution_count"], ev["analysis_id"]))

    # 8. 중복 등록 방지: import 를 여러 번 해도 이벤트는 셀당 1건
    before = len(RECEIVED)
    ip2.run_cell(store_history=True, raw_cell="import actverse\nimport actverse.telemetry\nfrom actverse.utils import load_json\nq = 1")
    time.sleep(0.7)
    assert len(RECEIVED) == before + 1, len(RECEIVED) - before
    print("8  no duplicate registration        OK  1 event per cell")

    # 9. ACTVERSE_TELEMETRY=0 → 전송 없음
    os.environ["ACTVERSE_TELEMETRY"] = "0"
    _purge_actverse()
    ip3 = _fresh_shell()
    before = len(RECEIVED)
    ip3.run_cell(store_history=True, raw_cell="import actverse\nw = 1")
    time.sleep(0.5)
    assert len(RECEIVED) == before
    del os.environ["ACTVERSE_TELEMETRY"]
    print("9  opt-out env                      OK  no events")

    # 10. IPython 밖 import → no-op
    out = subprocess.run(
        [sys.executable, "-c", "import actverse, actverse.telemetry as t; print(t.is_enabled())"],
        capture_output=True, text=True, cwd=str(REPO), timeout=60,
    )
    assert out.returncode == 0 and out.stdout.strip() == "False", out
    print("10 plain python no-op               OK")

    print(f"\nALL PASSED - {len(RECEIVED)} events received")


def test_telemetry_end_to_end():
    run_all()


if __name__ == "__main__":
    run_all()
