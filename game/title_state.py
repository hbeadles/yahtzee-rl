"""Title screen state for the Yahtzee RL game.

This module implements the title/main menu screen that is displayed when
the game starts.
"""

from typing import Optional
from game.state import State
from game import APP_SCREEN_WIDTH, APP_SCREEN_HEIGHT
from raylib import *
from game.base import Game


class TitleState(State):
    """Title screen state showing the main menu.

    This state displays the game title, welcome message, and a play button
    to start the game. It uses the game's shader for background effects.

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
        """Initialize the title state.

        Args:
            game: Reference to the main Game instance.
        """
        super().__init__(game)
        self.intro_label: bytes = b"Yahtzee RL Simulation"
        self.intro_box: bytes = b"Welcome to the Yahtzee RL Simulation!\nClick Play to start."
        self.shader: Optional[Shader] = None
        self.time_loc: int = -1
        self.res_loc: int = -1
        self.mouse_loc: int = -1
        self.play_button_rect: Optional[Rectangle] = None

    def on_enter(self) -> None:
        """Initialize shader references when entering the title state."""
        self.time_loc = self.game.time_loc
        self.res_loc = self.game.res_loc
        self.mouse_loc = self.game.mouse_loc
        self.shader = self.game.shader

        # Create play button rectangle
        button_width = 200
        button_height = 50
        button_x = APP_SCREEN_WIDTH // 2 - button_width // 2
        button_y = APP_SCREEN_HEIGHT // 2 + 100
        self.play_button_rect = ffi.new("struct Rectangle *",
                                        [button_x, button_y, button_width, button_height])[0]
        self.log(
            f"TitleState - resolution diag: screen={GetScreenWidth()}x{GetScreenHeight()} "
            f"render={GetRenderWidth()}x{GetRenderHeight()} "
            f"dpi_scale={GetWindowScaleDPI().x:.2f}x{GetWindowScaleDPI().y:.2f}")

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

        # Check for play button click
        if self.play_button_rect and GuiButton(self.play_button_rect, b"Play"):
            return "GAME"

        return None

    def render(self) -> None:
        """Render the title screen with shader background, title, and play button."""
        # Draw shader background
        if self.shader and IsShaderValid(self.shader):
            BeginShaderMode(self.shader)
            DrawRectangle(0, 0, APP_SCREEN_WIDTH, APP_SCREEN_HEIGHT, WHITE)
            EndShaderMode()

        # Draw title
        title_size = 48
        title_width = MeasureText(self.intro_label, title_size)
        title_x = APP_SCREEN_WIDTH // 2 - title_width // 2
        title_y = APP_SCREEN_HEIGHT // 4
        DrawText(self.intro_label, title_x, title_y, title_size, WHITE)

        # Draw welcome message
        message_size = 20
        message_lines = self.intro_box.split(b'\n')
        message_y = APP_SCREEN_HEIGHT // 2 - 50
        for line in message_lines:
            line_width = MeasureText(line, message_size)
            line_x = APP_SCREEN_WIDTH // 2 - line_width // 2
            DrawText(line, line_x, message_y, message_size, WHITE)
            message_y += 30

        # Play button is drawn by GuiButton in update()
        GuiButton(self.play_button_rect, b"Play")

