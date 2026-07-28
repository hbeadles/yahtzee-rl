"""Title screen state for the Yahtzee RL game.

This module implements the title/main menu screen that is displayed when
the game starts.
"""

from typing import Optional
from game.state import State
from game import APP_SCREEN_WIDTH, APP_SCREEN_HEIGHT
from raylib import *
from game.base import Game
import json
from pathlib import Path

class EndState(State):
    """End screen state showing the game results.

    This state displays the game results and a play button to restart the game.
    It uses the game's shader for background effects.

    Attributes:
        intro_label: Title text displayed at the top.
        intro_box: Welcome message text.
        shader: Reference to the game's shader.
        time_loc: Shader uniform location for time.
        res_loc: Shader uniform location for resolution.
        mouse_loc: Shader uniform location for mouse position.
        play_button_rect: Rectangle defining the play button bounds.
    """

    def __init__(self, game: Game) -> None:
        """Initialize the end state.

        Args:
            game: Reference to the main Game instance.
        """
        super().__init__(game)
        self.exit_label: bytes = b"Game Over"
        self.exit_width: int = MeasureText(self.exit_label, 48)
        self.score_label: bytes = b"Final Score: "
        self.yahtzee_label: bytes = b"Number of Yahtzees? "
        self.upper_score_label: bytes = b"Upper Score: "
        self.lower_score_label: bytes = b"Lower Score: "
        self.shader: Optional[Shader] = None
        self.time_loc: int = -1
        self.res_loc: int = -1
        self.mouse_loc: int = -1
        self.final_result_box: Optional[Rectangle] = None
        self.game_state_file: Path = Path("game_state_output.json")
        self.final_score = 0
        self.upper_score = 0
        self.lower_score = 0

    def on_enter(self) -> None:
        """Initialize shader references when entering the title state."""
        self.time_loc = self.game.time_loc
        self.res_loc = self.game.res_loc
        self.mouse_loc = self.game.mouse_loc
        self.shader = self.game.shader

        self.final_result_box = ffi.new("struct Rectangle *",
                                        [APP_SCREEN_WIDTH // 2 - 25, APP_SCREEN_HEIGHT // 2 + 250, 120, 50])[0]
        self.load_game_state()
    
    def load_game_state(self) -> None:
        """Load the final game state from a file."""
        if self.game_state_file.exists():
            with open(self.game_state_file, "r") as f:
                game_state = json.load(f)
                self.final_score = game_state.get("final_score", 0)
                self.upper_score = game_state.get("upper_score", 0)
                self.lower_score = game_state.get("lower_score", 0)
        else:
            self.final_score = 0
            self.upper_score = 0
            self.lower_score = 0
        


    def update(self, delta_time: float) -> Optional[str]:
        """Update the title state and check for play button click.

        Args:
            delta_time: Time elapsed since the last frame in seconds.

        Returns:
            "GAME" if the play button is clicked, None otherwise.
        """
        # Update shader uniforms
        if self.shader and IsShaderValid(self.shader):
            t = ffi.new("float *", GetTime())
            SetShaderValue(self.shader, self.time_loc, t, SHADER_UNIFORM_FLOAT)

            res = ffi.new("float[2]", [float(GetScreenWidth()), float(GetScreenHeight())])
            SetShaderValue(self.shader, self.res_loc, res, SHADER_UNIFORM_VEC2)

            mp = GetMousePosition()
            mouse = ffi.new("float[2]", [mp.x, float(GetScreenHeight()) - mp.y])
            SetShaderValue(self.shader, self.mouse_loc, mouse, SHADER_UNIFORM_VEC2)

        # # Check for play button click
        if self.final_result_box and GuiButton(self.final_result_box, b"Restart? Quit Otherwise"):
            return "TITLE"

        return None


    def render(self) -> None:
        """Render the title screen with shader background, title, and play button."""
        # Draw shader background
        if self.shader and IsShaderValid(self.shader):
            BeginShaderMode(self.shader)
            DrawRectangle(0, 0, APP_SCREEN_WIDTH, APP_SCREEN_HEIGHT, WHITE)
            EndShaderMode()

        message_x = APP_SCREEN_WIDTH // 2 - 75
        message_y = APP_SCREEN_HEIGHT // 2 - 150
        # Draw title
        DrawText(self.exit_label, message_x, message_y, 48, WHITE)

        gui_panel_x = APP_SCREEN_WIDTH // 2 - 50
        gui_panel_y = APP_SCREEN_HEIGHT // 2 - 50
        gui_panel_width = 200
        gui_panel_height = 220
        GuiPanel(ffi.new("struct Rectangle *", [gui_panel_x, gui_panel_y, gui_panel_width, gui_panel_height])[0], b"Final Results")
        GuiGroupBox(ffi.new("struct Rectangle *", [gui_panel_x + 10, gui_panel_y + 30, gui_panel_width - 20, gui_panel_height - 20])[0], b"Metrics")
        GuiLabel(ffi.new("struct Rectangle *", [gui_panel_x + 15, gui_panel_y + 60, gui_panel_width - 20, 25])[0], self.score_label + str(self.final_score).encode("utf-8"))
        GuiLabel(ffi.new("struct Rectangle *", [gui_panel_x + 15, gui_panel_y + 100, gui_panel_width - 20, 25])[0], self.upper_score_label + str(self.upper_score).encode("utf-8"))
        GuiLabel(ffi.new("struct Rectangle *", [gui_panel_x + 15, gui_panel_y + 130, gui_panel_width - 20, 25])[0], self.lower_score_label + str(self.lower_score).encode("utf-8"))

        # Play button is drawn by GuiButton in update()
        GuiButton(self.final_result_box, b"Restart?")

