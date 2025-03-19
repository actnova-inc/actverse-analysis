from pathlib import Path
from string import Template

import yaml


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


def create_notebook(lang, texts):
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


def build_templates(supported_languages):
    # build notebook for each language
    for lang in supported_languages:
        texts = load_yaml(lang)
        create_notebook(lang, texts)


if __name__ == "__main__":
    supported_languages = ["en", "ko"]
    build_templates(supported_languages)
