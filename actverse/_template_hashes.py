"""자동 생성 — bin/_templates/build.py 가 만든다. 직접 고치지 말 것.

템플릿 코드 셀의 정규화 해시 → 첫 코드 줄. actverse.telemetry 가 사용자
실행 셀을 original / modified / new 로 분류할 때 참조한다 (ACT-5939).
"""

TEMPLATE_CELL_HASHES = {
    '13609f5ed3218b582645a4362940c2bcf94af70a556f27be8667982d49188021': 'from actverse.utils import load_json',
    '1971f0ecde3c01de96f61b78363482d12e7cf89485e7166afb98cd69aeba3a68': 'mouse_id = ids[0]',
    '1eef149c08820956c05b1989176f1bc9394c094919c70603ea63df095abaf24a': 'import plotly.graph_objects as go',
    '35665337fa55193fdca70c98d568aa13d64dae6b963398939f7ecb7b9ff27b2e': 'mouse_id = ids[0]',
    '5ec80e6f9d7ec1b4c95c8f8feb6376a866ca5c8588c4936bce030fa13aa7508d': 'import numpy as np',
    '63fbc5ff93835fc28f8df9c42765e8c4b202cc2fa3fcdab56abfdebd549443be': 'from actverse.utils import load_json',
    '7e47647e82ddbf8cb6607d2558d5357afca2883afad951ea205ab3592d4058b4': 'import plotly.graph_objects as go',
    'a8afcb7db2fcbd4058915e2dd6b19c1b2c73860de6b11380caf13199299a580a': 'from actverse.utils.notebook import display_body_parts_checkbox',
    'b5dbe05aaf1516f9b265a016b9e1ce8d86b02e63eec2f8958c59c2f1b3c6fef0': 'img_width = prediction["metadata"]["origin_width"]',
    'bc21cfe52e35643226a7569cfb2c6765e255bac4cfb8cda42b3f87c9e334b4df': '! pip install -qq actverse@git+https://github.com/actnova-inc/actverse-analysis',
    'd27659f8fa6093753120442f0d5d9dfed2049ed46b5bccf50975c09d739d6d4b': 'import numpy as np',
    'e4e3ed96662e56a139b82cf4a5d3a936dd1d2d3ac60a92f77aeabb62f3a9b7b9': 'from actverse.utils.notebook import display_body_parts_checkbox',
    'f7e2e32598727895e2018ad542a108b514458b083bd343757a0e6f3787e6e0da': 'from actverse.analysis import measure_physical_metrics',
}
