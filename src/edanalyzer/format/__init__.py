import rich
from rich import print as rprint
from rich.panel import Panel
from rich.align import Align
from rich.padding import Padding

def indent_text(text, indent=4):
    return Padding(
        text,
        (0, 0, 0, indent)
    )