# ACT-5939: Colab 셀 실행 텔레메트리. 어떤 하위 모듈이 import 되든 패키지
# 초기화가 먼저 돌므로 여기서 등록한다. IPython 밖에서는 no-op.
try:
    from . import telemetry as _telemetry  # noqa: F401
except Exception:  # 텔레메트리 문제로 분석 기능이 막히면 안 된다
    pass
