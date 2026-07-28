"""Gameplay state for the Yahtzee RL game.

This module implements the main gameplay screen where the user can play Yahtzee,
roll dice, and select scoring categories.
"""

from typing import Optional, Dict, Any, List
from game.state import State
from game import APP_SCREEN_WIDTH, APP_SCREEN_HEIGHT
from raylib import *
from game.base import Game
from game import (YahtzeeGameStates,
                  DiceAnimationState,
                  ScorecardGameStates)
from yahtzee_rl.config import CATEGORY_NAMES
from yahtzee_rl.envs.yahtzee_env import YahtzeeEnv
from collections import defaultdict
from game.objects.dice_roll_object import DiceRollObject
from game.objects.score_card_object import ScoreCardObject
from game.objects.game_info_overlay import GameInfoOverlayObject
from game.objects.guide_text_object import GuideTextObject
from game.objects.ai_assist_button import AiAssistButton
from yahtzee_rl.config import Category
from pathlib import Path
import json

GUIDE_MESSAGES: dict[YahtzeeGameStates, str] = {
    YahtzeeGameStates.GAME_DICE_ROLLING_ACTION: "Rolling dice...",
    YahtzeeGameStates.GAME_DICE_SELECTION_ACTION: "Select dice to reroll by left-clicking on them, then press Reroll.",
    YahtzeeGameStates.GAME_SCORECARD_SELECTION_ACTION: "Choose a scoring category.",
    YahtzeeGameStates.GAME_ROUND_END: "Round complete! Starting next round...",
    YahtzeeGameStates.GAME_ENDED: "Game over! Final score: %d",
}

class GameState(State):

    def __init__(self,
                 game: Game):
        super().__init__(game)
        self.yahtzee_env = YahtzeeEnv()
        self.observation = None
        self.parsed_observation: dict[str, Any] = None
        self.dice: List[Any] = []
        self.game_time = None
        self.yahtzee_game_state = YahtzeeGameStates.GAME_START
        self.scorecard: Dict[str|Category, int] = {}
        self.upper_score: int = 0
        self.lower_score: int = 0
        self.total_score: int = 0
        self.game_over: bool = False
        self.shader = None
        self.round_number = 0
        self.roll_number = 0
        self.time_loc = -1
        self.mouse_loc = -1
        self.res_loc = -1

    def on_enter(self) -> None:
        """Initialize resources when entering the gameplay state."""
        self.game_time = GetTime()
        self.shader = self.game.shader
        self.time_loc = self.game.time_loc
        self.mouse_loc = self.game.mouse_loc
        self.res_loc = self.game.res_loc
        self.observation, _ = self.yahtzee_env.reset()
        self.parsed_observation = YahtzeeEnv.parse_observation(self.observation)
        self.dice = self.parsed_observation['dice']


class GameStateDeploy(GameState):
    """Main gameplay state for Yahtzee.
    
    This state handles the core game loop including dice rolling, scorecard
    interaction, and game rendering.
    
    Attributes:
        category_scores: List tracking which categories have been scored.
        category_names: Encoded category names for display.
        game_time: Current game time.
    """

    def __init__(self, game: Game) -> None:
        """Initialize the gameplay state.
        
        Args:
            game: Reference to the main Game instance.
        """
        super().__init__(game)
        self.dice_animations: Dict[int, DiceAnimationState] = defaultdict()
        self.scorecard_game_state = [
            ScorecardGameStates.SCORECARD_IDLE
            for _ in range(len(CATEGORY_NAMES))
        ]
        self.dice_row = None
        self.info_overlay = None
        self.guide_text = None
        self.gui_button_texture_obj = None
        self._prev_yahtzee_state = None
        self.output_path: Path = Path("game_state_output.json")


    def on_enter(self) -> None:
        """Initialize resources when entering the gameplay state."""
        self.game_time = GetTime()
        self.shader = self.game.shader
        self.time_loc = self.game.time_loc
        self.mouse_loc = self.game.mouse_loc
        self.res_loc = self.game.res_loc
        self.yahtzee_game_state = YahtzeeGameStates.GAME_START
        self.game_over = False
        self.observation, _ = self.yahtzee_env.reset()
        self.parsed_observation = YahtzeeEnv.parse_observation(self.observation)
        self.dice = self.parsed_observation['dice']
        self.scorecard = self.parsed_observation['scorecard']
        self.round_number = self.parsed_observation['round']
        self.roll_number = self.parsed_observation['rolls_remaining']
        self.upper_score = self.parsed_observation['upper_score']
        self.lower_score = self.parsed_observation['lower_score']
        self.total_score = self.upper_score + self.lower_score
        self.dice_row = DiceRollObject(name="dice_row", state=self, game=self.game)
        self.info_overlay = GameInfoOverlayObject(name="info_overlay", state=self, game=self.game)
        self.guide_text = GuideTextObject(name="guide_text", state=self, game=self.game)
        self.scorecard_render_obj = ScoreCardObject(name="scorecard", state=self, game=self.game,
                                                    scorecard=self.parsed_observation['scorecard'])
        self.gui_button_texture_obj = AiAssistButton(name="gui_button_texture_obj", 
                                        state=self, game=self.game, button_text=b"Ask AI for help!",
                                        width=110, height=40)
        
    def on_exit(self) -> None:
        """Cleanup resources when exiting the gameplay state."""
        self.dice_row = None
        self.info_overlay = None
        self.guide_text = None
        self.scorecard_render_obj = None
        self.gui_button_texture_obj = None
        self.shader = None
        self.dice = None
        self.round_number = 0
        self.roll_number = 0
        self.upper_score = 0
        self.lower_score = 0
        self.total_score = 0


    def update(self, delta_time: float) -> Optional[str]:
        """Update the gameplay state.
        
        Args:
            delta_time: Time elapsed since the last frame in seconds.
        
        Returns:
            Name of the next state to transition to, or None to stay in gameplay.
        """
        self.game_time = GetTime()
        # Update shader uniforms
        if self.shader and IsShaderValid(self.shader):
            t = ffi.new("float *", GetTime())
            SetShaderValue(self.shader, self.time_loc, t, SHADER_UNIFORM_FLOAT)
            
            res = ffi.new("float[2]", [float(GetScreenWidth()), float(GetScreenHeight())])
            SetShaderValue(self.shader, self.res_loc, res, SHADER_UNIFORM_VEC2)
            
            mp = GetMousePosition()
            mouse = ffi.new("float[2]", [mp.x, float(GetScreenHeight()) - mp.y])
            SetShaderValue(self.shader, self.mouse_loc, mouse, SHADER_UNIFORM_VEC2)

        prev_state = self.yahtzee_game_state

        if self.yahtzee_game_state == YahtzeeGameStates.GAME_START:
            self.parsed_observation = YahtzeeEnv.parse_observation(self.observation)
            self.dice = self.parsed_observation['dice']
            self.round_number = self.parsed_observation['round']
            self.roll_number = self.parsed_observation['rolls_remaining']
            self.scorecard = self.parsed_observation['scorecard']
            self.yahtzee_game_state = YahtzeeGameStates.GAME_DICE_ROLLING_ACTION
            self.start_delay = 0.2
            self.dice_row.start_roll(self.dice, self.start_delay)
        elif self.yahtzee_game_state == YahtzeeGameStates.GAME_DICE_ROLLING_ACTION:
            if self.dice_row.all_ready():
                if self.roll_number > 0:
                    self.yahtzee_game_state = YahtzeeGameStates.GAME_DICE_SELECTION_ACTION
                else:
                    self.yahtzee_game_state = YahtzeeGameStates.GAME_SCORECARD_SELECTION_ACTION
            
        elif self.yahtzee_game_state == YahtzeeGameStates.GAME_DICE_SELECTION_ACTION and self.roll_number > 0:
            if self.dice_row.roll_requested:
                self.dice_row.roll_requested = False
                selected_mask = self.dice_row.selected_mask()
                # Take a step
                self.observation, reward, done, truncated, info = self.yahtzee_env.step(selected_mask)
                self.game_over = done
                self.parsed_observation = YahtzeeEnv.parse_observation(self.observation)
                self.dice = self.parsed_observation['dice']
                self.scorecard = self.parsed_observation['scorecard']
                self.round_number = self.parsed_observation['round']
                self.roll_number = self.parsed_observation['rolls_remaining']
                self.upper_score = self.parsed_observation['upper_score']
                self.lower_score = self.parsed_observation['lower_score']
                self.total_score = self.upper_score + self.lower_score
                self.dice_row.start_roll(self.dice, self.start_delay)
                self.yahtzee_game_state = YahtzeeGameStates.GAME_DICE_ROLLING_ACTION
        elif self.yahtzee_game_state == YahtzeeGameStates.GAME_SCORECARD_SELECTION_ACTION:
            if self.scorecard_render_obj.scorecard_selected:
                self.scorecard_render_obj.scorecard_selected = False
                category_action = self.scorecard_render_obj.get_category_action()
                self.observation, reward, done, truncated, info = self.yahtzee_env.step(category_action)
                self.game_over = done
                self.parsed_observation = YahtzeeEnv.parse_observation(self.observation)
                self.dice = self.parsed_observation['dice']
                self.round_number = self.parsed_observation['round']
                self.scorecard = self.parsed_observation['scorecard']
                self.roll_number = self.parsed_observation['rolls_remaining']
                self.scorecard = self.parsed_observation['scorecard']
                self.upper_score = self.parsed_observation['upper_score']
                self.lower_score = self.parsed_observation['lower_score']
                self.total_score = self.upper_score + self.lower_score
                self.yahtzee_game_state = YahtzeeGameStates.GAME_ROUND_END
                self.scorecard_render_obj.reset()

        elif self.yahtzee_game_state == YahtzeeGameStates.GAME_ROUND_END:
            if self.game_over:
                self.yahtzee_game_state = YahtzeeGameStates.GAME_ENDED
                self.parsed_observation = YahtzeeEnv.parse_observation(self.observation)
                self.upper_score = self.parsed_observation['upper_score']
                self.lower_score = self.parsed_observation['lower_score']
                self.total_score = self.upper_score + self.lower_score
                self.save_game_state()
                self.game.state_manager.change_state("END")
                return
                #self.guide_text.show_message(f"Game over! Final score: {self.total_score}")
            else:
                self.yahtzee_game_state = YahtzeeGameStates.GAME_START

        if self.yahtzee_game_state != prev_state:
            msg = GUIDE_MESSAGES.get(self.yahtzee_game_state)
            if msg:
                template = GUIDE_MESSAGES[self.yahtzee_game_state]
                message = template(self.total_score) if callable(template) else template
                if (self.yahtzee_game_state == YahtzeeGameStates.GAME_SCORECARD_SELECTION_ACTION
                        and self.yahtzee_env.game.scorecard.joker_active(self.yahtzee_env.game.dice)):
                    message = "JOKER! You rolled another Yahtzee. Score it in any open category."
                self.guide_text.show_message(message)

        self.dice_row.update(delta_time)
        self.info_overlay.update(delta_time)
        self.guide_text.update(delta_time)
        self.scorecard_render_obj.update(delta_time)
        self.gui_button_texture_obj.update(delta_time)
        return None

    def render(self) -> None:
        """Render the gameplay state."""
        if IsShaderValid(self.shader):
            BeginShaderMode(self.shader)
            DrawRectangle(0, 0, APP_SCREEN_WIDTH, APP_SCREEN_HEIGHT, WHITE)
            EndShaderMode()
        self.scorecard_render_obj.render()
        #self.render_scorecard()
        DrawText(b"Yahtzee", self.game.screen_width // 2, 20, 28, WHITE)
        self.dice_row.render()
        self.guide_text.render()
        self.info_overlay.render()
        self.gui_button_texture_obj.render()

    def save_game_state(self) -> None:
        """Save the current game state to a file."""
        game_final_state = {
            "final_score": self.total_score,
            "upper_score": self.upper_score,
            "lower_score": self.lower_score
        }

        with open(self.output_path, "w") as f:
            json.dump(game_final_state, f, indent=4)
