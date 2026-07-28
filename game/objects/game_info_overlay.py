from raylib import *
from game.objects.game_object import GameObject
from game.components.slide_component import SlideComponent
from game.base import Game

PANEL_WIDTH = 300
TAB_WIDTH = 30
PADDING = 15

GAME_FLOW_STEPS = [
    b"1. Roll all 5 dice.",
    b"2. Choose dice to keep, then",
    b"   reroll the rest (2 chances).",
    b"3. Pick a scoring category.",
]

UPPER_CATEGORIES = [
    (b"Ones", b"Sum of all ones"),
    (b"Twos", b"Sum of all twos"),
    (b"Threes", b"Sum of all threes"),
    (b"Fours", b"Sum of all fours"),
    (b"Fives", b"Sum of all fives"),
    (b"Sixes", b"Sum of all sixes"),
]

LOWER_CATEGORIES = [
    (b"3 of a Kind", b"Sum of all dice (3+ match)"),
    (b"4 of a Kind", b"Sum of all dice (4+ match)"),
    (b"Full House", b"25 pts (3+2 of a kind)"),
    (b"Sm. Straight", b"30 pts (4 in sequence)"),
    (b"Lg. Straight", b"40 pts (5 in sequence)"),
    (b"Yahtzee", b"50 pts (all 5 match)"),
    (b"Chance", b"Sum of all dice"),
]


class GameInfoOverlayObject(GameObject):

    def __init__(self, name, state, game):
        super().__init__(name, state, game)
        self.add(SlideComponent(
            name="slide",
            game=game,
            closed_pos=float(game.screen_width),
            open_pos=float(game.screen_width - PANEL_WIDTH),
        ))

    @property
    def slide(self) -> SlideComponent:
        return self.get("slide")

    def render(self):
        slide = self.slide
        x = int(slide.current_pos)
        h = self.game.screen_height

        tab_x = x - TAB_WIDTH
        tab_y = h // 2 - 20
        tab_rect = ffi.new("struct Rectangle *",
                           [tab_x, tab_y, TAB_WIDTH, 40])[0]
        label = b"<<" if slide.is_open else b">>"
        if GuiButton(tab_rect, label):
            slide.toggle()

        if slide.is_settled and not slide.is_open:
            return

        GuiSetAlpha(0.90)
        GuiPanel(ffi.new("struct Rectangle *",
                         [x, 0, PANEL_WIDTH, h])[0], b"How to Play")
        GuiSetAlpha(1.0)

        needs_lock = not slide.is_open and not slide.is_settled
        if needs_lock:
            GuiLock()

        cx = x + PADDING
        cy = 30
        content_w = PANEL_WIDTH - PADDING * 2

        # --- Game Flow ---
        DrawText(b"Game Flow", cx, cy, 18, DARKPURPLE)
        cy += 25
        DrawLine(cx, cy, cx + content_w, cy, GRAY)
        cy += 8
        for step in GAME_FLOW_STEPS:
            DrawText(step, cx, cy, 14, GRAY)
            cy += 18
        cy += 10

        # --- Upper Section ---
        DrawText(b"Upper Section", cx, cy, 18, DARKPURPLE)
        cy += 25
        DrawLine(cx, cy, cx + content_w, cy, GRAY)
        cy += 8
        for name, desc in UPPER_CATEGORIES:
            DrawText(name, cx, cy, 14, MAROON)
            DrawText(desc, cx + 70, cy, 12, GRAY)
            cy += 20
        cy += 5
        DrawText(b"Bonus: +35 pts if upper >= 63",
                 cx, cy, 13, DARKGRAY)
        cy += 25

        # --- Lower Section ---
        DrawText(b"Lower Section", cx, cy, 18, DARKPURPLE)
        cy += 25
        DrawLine(cx, cy, cx + content_w, cy, GRAY)
        cy += 8
        for name, desc in LOWER_CATEGORIES:
            DrawText(name, cx, cy, 14, MAROON)
            DrawText(desc, cx + 110, cy, 12, GRAY)
            cy += 20
        cy += 5
        DrawText(b"Yahtzee Bonus: +100 per extra",
                 cx, cy, 13, DARKGRAY)

        if needs_lock:
            GuiUnlock()
