"""启动页的像素字标：用色块拼出一个 N，不依赖图片或终端图形协议。"""

from __future__ import annotations

from rich.align import Align
from rich.text import Text

PALETTE = {
    "T": "#50b7c2",
    "t": "#2f7f88",
    "D": "#a6d189",
    "G": "#7bba55",
    "g": "#4f8a3a",
    "W": "#e6e7ea",
}

# 每个字符是一个逻辑像素；渲染时扩成两个终端格，视觉上接近正方形。
HERO_GRID: tuple[str, ...] = (
    "WWWWDDD............WWWW",
    "TTTT.DDD...........GGGG",
    "TTTT..DDD..........GGGG",
    "TTTT...DDD.........GGGG",
    "TTTT...DDD.........GGGG",
    "TTTT....DDD........GGGG",
    "TTTT.....DDD.......GGGG",
    "TTTT......DDD......GGGG",
    "TTTT.......DDD.....GGGG",
    "TTTT........DDD....GGGG",
    "TTTT.........DDD...GGGG",
    "TTTT.........DDD...GGGG",
    "TTTT..........DDD..GGGG",
    "TTTT...........DDD.GGGG",
    "tttt............DDDgggg",
)

MEDIUM_GRID: tuple[str, ...] = (
    "WWWDD.....WWW",
    "TTT.DD....GGG",
    "TTT..DD...GGG",
    "TTT..DD...GGG",
    "TTT...DD..GGG",
    "TTT....DD.GGG",
    "ttt.....DDggg",
)


def tier_for(width: int, height: int) -> str:
    """按终端尺寸选择字标档位；阈值保证弹窗不会被撑破。"""
    if width >= 72 and height >= 34:
        return "hero"
    if width >= 60 and height >= 24:
        return "medium"
    return "compact"


def _grid_text(grid: tuple[str, ...], *, trailing_blank: bool = False) -> Text:
    text = Text()
    text.no_wrap = True
    for row_index, row in enumerate(grid):
        for pixel in row:
            if pixel == ".":
                text.append("  ")
            else:
                text.append("██", style=PALETTE[pixel])
        if row_index < len(grid) - 1:
            text.append("\n")
    if trailing_blank:
        text.append("\n")
    return text


def _wordmark() -> Text:
    text = Text()
    text.no_wrap = True
    text.append(">", style="bold #50b7c2")
    text.append("_", style="bold #e6e7ea")
    text.append("  NextCoder", style="bold #7bba55")
    return text


def welcome_renderable(width: int, height: int) -> Align:
    """返回居中的字标；窄窗口自动退回单行文字。"""
    tier = tier_for(width, height)
    if tier == "hero":
        return Align.center(_grid_text(HERO_GRID, trailing_blank=True))
    if tier == "medium":
        return Align.center(_grid_text(MEDIUM_GRID))
    return Align.center(_wordmark())
