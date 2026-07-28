from raylib import *
from game.components.component import Component
from game.base import Game


class TypewriterComponent(Component):
    """Reveals text one character at a time with word-wrap support."""

    def __init__(self, name: str, game: Game, chars_per_second: float = 30.0):
        super().__init__(name, game)
        self._full_text: str = ""
        self._lines: list[str] = []
        self._char_index: float = 0.0
        self._cps: float = chars_per_second
        self._finished: bool = True

    @property
    def visible_lines(self) -> list[bytes]:
        count = min(int(self._char_index), len(self._full_text))
        remaining = count
        result: list[bytes] = []
        for line in self._lines:
            if remaining <= 0:
                break
            visible_count = min(remaining, len(line))
            result.append(line[:visible_count].encode("utf-8"))
            remaining -= visible_count
            # The space consumed by the line break
            remaining -= 1
        return result

    @property
    def is_finished(self) -> bool:
        return self._finished

    @property
    def line_count(self) -> int:
        return len(self._lines)

    def set_text(self, text: str, font_size: int, max_width: int):
        self._full_text = text
        self._lines = self._wrap_text(text, font_size, max_width)
        self._char_index = 0.0
        self._finished = False

    def skip(self):
        self._char_index = float(len(self._full_text))
        self._finished = True

    def update(self, delta_time: float):
        if self._finished:
            return
        self._char_index += self._cps * delta_time
        if int(self._char_index) >= len(self._full_text):
            self._char_index = float(len(self._full_text))
            self._finished = True

    def _wrap_text(self, text: str, font_size: int, max_width: int) -> list[str]:
        all_lines: list[str] = []
        for paragraph in text.split("\n"):
            words = paragraph.split(" ")
            current_line = ""
            for word in words:
                test = f"{current_line} {word}".strip()
                if MeasureText(test.encode("utf-8"), font_size) > max_width:
                    if current_line:
                        all_lines.append(current_line)
                    current_line = word
                else:
                    current_line = test
            if current_line:
                all_lines.append(current_line)
        return all_lines
