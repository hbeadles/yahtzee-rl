from collections import deque

from raylib import *
from game.objects.game_object import GameObject
from game.components.typewriter_component import TypewriterComponent

GUIDE_PANEL_WIDTH = 400
GUIDE_PADDING = 25
GUIDE_FONT_SIZE = 18
GUIDE_LINE_HEIGHT = GUIDE_FONT_SIZE + 4


class GuideTextObject(GameObject):
    """Semi-transparent text panel below the dice area with typewriter reveal."""

    def __init__(self, name, state, game,
                 chars_per_second: float = 30.0,
                 hold_time: float = 1.0):
        super().__init__(name, state, game)
        self._message_queue: deque[str] = deque()
        self._hold_timer: float = 0.0
        self._hold_time: float = hold_time
        self._holding: bool = False
        self.add(TypewriterComponent(
            name="typewriter",
            game=game,
            chars_per_second=chars_per_second,
        ))

    @property
    def typewriter(self) -> TypewriterComponent:
        return self.get("typewriter")

    def show_message(self, text: str):
        if self.typewriter.is_finished and not self._holding:
            self._start_message(text)
        else:
            self._message_queue.append(text)

    def clear_queue(self):
        self._message_queue.clear()
        self._holding = False
        self._hold_timer = 0.0

    def _start_message(self, text: str):
        content_w = GUIDE_PANEL_WIDTH - GUIDE_PADDING * 2
        self.typewriter.set_text(text, GUIDE_FONT_SIZE, content_w)
        self._holding = False
        self._hold_timer = 0.0

    def update(self, delta_time: float):
        super().update(delta_time)

        if self._holding:
            self._hold_timer += delta_time
            if self._hold_timer >= self._hold_time and self._message_queue:
                self._start_message(self._message_queue.popleft())
            return

        if self.typewriter.is_finished and self._message_queue:
            self._holding = True
            self._hold_timer = 0.0

    def render(self):
        tw = self.typewriter
        lines = tw.visible_lines
        if not lines:
            return

        panel_x = self.game.screen_width // 2 - 85
        panel_y = self.game.screen_height // 2 + 100
        panel_h = GUIDE_PADDING * 2 + GUIDE_LINE_HEIGHT * tw.line_count

        GuiSetAlpha(0.95)
        GuiPanel(
            ffi.new("struct Rectangle *",
                     [panel_x, panel_y, GUIDE_PANEL_WIDTH, panel_h])[0],
            b"Guide",
        )
        GuiSetAlpha(1.0)

        text_x = panel_x + GUIDE_PADDING
        text_y = panel_y + GUIDE_PADDING
        for line_bytes in lines:
            DrawText(line_bytes, text_x, text_y, GUIDE_FONT_SIZE, BLACK)
            text_y += GUIDE_LINE_HEIGHT
