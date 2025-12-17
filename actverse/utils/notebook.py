import ipywidgets as widgets
from IPython.display import display

from actverse.entity import BodyPart
from actverse.entity.mouse import BODY_PART_INDEX as MOUSE_BODY_PART_INDEX

_cached_checkboxes = None


def display_body_parts_checkbox(
    description: str = "Select the body parts to analyze", lang: str = "en"
):
    global _cached_checkboxes
    body_parts = MOUSE_BODY_PART_INDEX.keys()

    label = widgets.Label(description)
    if _cached_checkboxes is None:
        checkboxes = []
        for body_part in body_parts:
            body_part.set_lang(lang)
            checkboxes.append(widgets.Checkbox(value=False, description=str(body_part)))
        checkboxes[-1].disabled = False
        checkboxes[-1].value = True
    else:
        checkboxes = _cached_checkboxes

    display(
        label,
        *checkboxes,
    )
    _cached_checkboxes = checkboxes
    return checkboxes


def get_checked(checkboxes: list[widgets.Checkbox]) -> list[BodyPart]:
    body_parts = MOUSE_BODY_PART_INDEX.keys()

    checked = []
    for checkbox, body_part in zip(checkboxes, body_parts):
        if checkbox.value:
            checked.append(body_part)
    return checked
