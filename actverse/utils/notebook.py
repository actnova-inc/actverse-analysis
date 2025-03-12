import ipywidgets as widgets
from IPython.display import display

from actverse.entity.mouse import BODY_PART_INDEX as MOUSE_BODY_PART_INDEX


def display_body_parts_checkbox(
    description: str = "Select the body parts to analyze",
):
    body_parts = MOUSE_BODY_PART_INDEX.keys()

    label = widgets.Label(description)
    checkboxes = [
        widgets.Checkbox(value=False, description=part) for part in body_parts
    ]
    checkboxes[-1].disabled = True
    checkboxes[-1].value = True

    display(
        label,
        *checkboxes,
    )
    return checkboxes


def get_checked(checkboxes: list[widgets.Checkbox]) -> list[str]:
    checked = []
    for checkbox in checkboxes:
        if checkbox.value:
            checked.append(checkbox.description)
    return checked
