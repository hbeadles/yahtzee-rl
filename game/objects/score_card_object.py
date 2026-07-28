from game.objects.game_object import GameObject
from game.base import Game
from game.state import State
from yahtzee_rl.config import Category, CATEGORY_TO_ACTION, LOWER_CATEGORY_NAMES, UPPER_CATEGORY_NAMES
from typing import Dict
from game import ScorecardGameStates
from raylib import *

class ScoreCardObject(GameObject):
    """
    Object that manages the score card state and UI.
    Need to synchronize the state of the actual score card in the env
    with this object. This one should purely be a UI object.
    """

    def __init__(self, name, 
                 state: State, 
                 game: Game,
                 scorecard: Dict[str|Category, int]):
        super().__init__(name, state, game)
        self.scorecard: Dict[str|Category, int] = {}
        self.category_action: int = CATEGORY_TO_ACTION[Category.ACES]
        self.scorecard_selected: bool = False
        self.upper_score: int = 0
        self.lower_score: int = 0
        self.total_score: int = 0
        self.scorecard_state_map: Dict[str|Category, ScorecardGameStates] = {}
        for category in Category:
            self.scorecard[category] = 0
            self.scorecard_state_map[category] = ScorecardGameStates.SCORECARD_IDLE
        self.original_base_color_disabled = GuiGetStyle(BUTTON, BASE_COLOR_DISABLED)
        self.original_base_color_normal = GuiGetStyle(BUTTON, BASE_COLOR_NORMAL)

    
    def update(self, delta_time: float):
        self.scorecard = self.state.scorecard
        self.upper_score = self.state.upper_score
        self.lower_score = self.state.lower_score
        self.total_score = self.total_score

    def get_category_action(self) -> int:
        return self.category_action

    def reset(self):
        self.category_action: int = CATEGORY_TO_ACTION[Category.ACES]
        self.scorecard_selected = False
        for category in Category:
            self.scorecard_state_map[category] = ScorecardGameStates.SCORECARD_IDLE

    def render(self):
        GuiPanel(ffi.new("struct Rectangle *", [0, 0, 260, self.game.screen_height])[0], b"Scorecard")
        GuiGroupBox(ffi.new("struct Rectangle *", [10, 40, 240, 200])[0], b"Upper Section")
        y = 55
        rolls_remain = self.state.roll_number > 0
        for category in Category.upper_categories():
            rect = ffi.new("struct Rectangle *", [20, y, 220, 25])[0]
            locked = self.scorecard_state_map[category] == ScorecardGameStates.SCORECARD_LOCKED
            marked = self.scorecard[category] > 0
            if marked:
                GuiDisable()
                GuiSetStyle(BUTTON, BASE_COLOR_DISABLED, ColorToInt(DARKGRAY))
            if locked or rolls_remain:
                GuiDisable()
                if marked:
                    GuiSetStyle(BUTTON, BASE_COLOR_DISABLED, ColorToInt(DARKGRAY))
            if GuiButton(rect, category.name.encode("utf-8")):
                
                self.category_action = CATEGORY_TO_ACTION[category]
                self.scorecard_selected = True
                self.state.guide_text.show_message(f"Category {category.name} chosen")
                self.scorecard_state_map[category] = ScorecardGameStates.SCORECARD_LOCKED
            GuiEnable()
            GuiSetStyle(BUTTON, BASE_COLOR_DISABLED, self.original_base_color_disabled)
            y += 28
        GuiGroupBox(ffi.new("struct Rectangle *", [10, 255, 240, 220])[0], b"Lower Section")
        y = 275
        for category in Category.lower_categories():
            rect = ffi.new("struct Rectangle *", [20, y, 220, 25])[0]
            locked = self.scorecard_state_map[category] == ScorecardGameStates.SCORECARD_LOCKED
            marked = self.scorecard[category] > 0
            if marked:
                GuiDisable()
                GuiSetStyle(BUTTON, BASE_COLOR_DISABLED, ColorToInt(DARKGRAY))
            if locked or rolls_remain:
                GuiDisable()
                if marked:
                    GuiSetStyle(BUTTON, BASE_COLOR_DISABLED, ColorToInt(DARKGRAY))
            if GuiButton(rect, category.name.encode("utf-8")):
                self.category_action = CATEGORY_TO_ACTION[category]
                self.scorecard_selected = True
                self.state.guide_text.show_message(f"Category {category.name} chosen")
                self.scorecard_state_map[category] = ScorecardGameStates.SCORECARD_LOCKED
            GuiEnable()
            GuiSetStyle(BUTTON, BASE_COLOR_DISABLED, self.original_base_color_disabled)
            y += 28

        GuiGroupBox(ffi.new("struct Rectangle *", [10, 600, 150, 130])[0], b"Final Scores")
        upper_score_label = f"Upper Score: {self.upper_score}"
        lower_score_label = f"Lower Score: {self.lower_score}"
        total_score_label = f"Total Score: {self.total_score}"
        GuiLabel(ffi.new("struct Rectangle *", [20, 620, 130, 25])[0], upper_score_label.encode("utf-8"))
        GuiLabel(ffi.new("struct Rectangle *", [20, 650, 130, 25])[0], lower_score_label.encode("utf-8"))
        GuiLabel(ffi.new("struct Rectangle *", [20, 680, 130, 25])[0], total_score_label.encode("utf-8"))

    
