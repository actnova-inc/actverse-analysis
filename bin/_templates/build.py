import hashlib
import json
import sys
from pathlib import Path
from string import Template

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from actverse.telemetry import normalize_source  # noqa: E402


def flatten_yaml(lang_yaml):
    for key, value in lang_yaml.items():
        if isinstance(value, dict):
            yield from flatten_yaml(value)
        else:
            yield key, value


def load_yaml(lang: str) -> dict:
    # load lang yaml file
    yaml_path = Path(__file__).parent / "lang" / f"{lang}.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        return dict(flatten_yaml(yaml.safe_load(f)))


def create_notebook(lang, texts) -> Path:
    # create notebook file using template
    template_path = Path(__file__).parent / "custom_analysis.ipynb.template"
    output_dir = Path(__file__).parent.parent / lang
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "custom_analysis.ipynb"

    # load template notebook
    with open(template_path, "r", encoding="utf-8") as f:
        notebook_template = Template(f.read())

    # fill template and save
    notebook_template = notebook_template.substitute(texts)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(notebook_template)
    return output_path


def _first_code_line(raw: str) -> str:
    for line in raw.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            return s
    return ""


def collect_cell_hashes(notebook_paths) -> dict:
    """ACT-5939: 빌드된 노트북의 코드 셀 해시 → 첫 코드 줄.

    actverse.telemetry 가 사용자 실행 셀을 original / modified / new 로
    분류할 때 쓴다. 해시는 telemetry.normalize_source 와 같은 정규화를 거친다.
    서버가 치환하는 줄(ACTVERSE_*_ID, json_path)은 normalize_source 가 좌변만
    남기므로 치환 전후 해시가 같다.
    """
    hashes = {}
    for path in notebook_paths:
        nb = json.loads(Path(path).read_text(encoding="utf-8"))
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            h = hashlib.sha256(normalize_source(src).encode("utf-8")).hexdigest()
            hashes[h] = _first_code_line(src)
    return hashes


def write_template_hashes(hashes: dict):
    out = REPO_ROOT / "actverse" / "_template_hashes.py"
    lines = [
        '"""자동 생성 — bin/_templates/build.py 가 만든다. 직접 고치지 말 것.',
        "",
        "템플릿 코드 셀의 정규화 해시 → 첫 코드 줄. actverse.telemetry 가 사용자",
        "실행 셀을 original / modified / new 로 분류할 때 참조한다 (ACT-5939).",
        '"""',
        "",
        "TEMPLATE_CELL_HASHES = {",
    ]
    for h, first in sorted(hashes.items()):
        lines.append(f"    {h!r}: {first!r},")
    lines.append("}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def build_templates(supported_languages):
    # build notebook for each language
    outputs = []
    for lang in supported_languages:
        texts = load_yaml(lang)
        outputs.append(create_notebook(lang, texts))
    hashes = collect_cell_hashes(outputs)
    path = write_template_hashes(hashes)
    print(f"built {len(outputs)} notebooks, {len(hashes)} template cell hashes -> {path}")


if __name__ == "__main__":
    supported_languages = ["en", "ko"]
    build_templates(supported_languages)
