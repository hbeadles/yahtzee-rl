"""Game module for Yahtzee RL simulation.

This module provides the main Game class that manages the Yahtzee game state,
rendering, and shader processing using the raylib library.
"""

from typing import Dict, Optional
from raylib import *
from game import (APP_SCREEN_WIDTH,
                  APP_SCREEN_HEIGHT, DEFAULT_STYLE,
                  DICE_TEXTURE_MAP)
from collections import defaultdict
import re
import platform
from pathlib import Path
from game.state import StateManager
import asyncio


class Game:
    """Main game class for Yahtzee RL simulation.

    This class manages the game window, shaders, textures, and game state
    for the Yahtzee reinforcement learning simulation. It handles initialization
    of raylib components and provides state management for the game.

    Attributes:
        title: Window title as bytes string.
        base_shader_path: Path to the base fragment shader file.
        gui_style: Path to the GUI style file as bytes string.
        yahtzee_game_state: Current state of the Yahtzee game.
        dice_game_states: List of states for each of the 5 dice.
        scorecard_game_states: List of states for each scorecard category.
        config_flags: Raylib configuration flags for the window.
        time_loc: Shader uniform location for time.
        res_loc: Shader uniform location for resolution.
        mouse_loc: Shader uniform location for mouse position.
        shader: Loaded shader object.
        texture_mapping: Dictionary mapping texture names to loaded textures.
        fps: Target frames per second.
        game_time: Current game time in seconds.
        state_manager: StateManager instance for managing game screens.
    """

    def __init__(self,
                 base_shader_path: Path,
                 title: bytes = b"Yahtzee RL Simulation",
                 gui_style: bytes = DEFAULT_STYLE,
                 target_fps: int = 60
                 ) -> None:
        """Initialize the Game instance.

        Args:
            base_shader_path: Path to the base fragment shader file.
            title: Window title as bytes string. Defaults to b"Yahtzee RL Simulation".
            gui_style: Path to the GUI style file as bytes string. Defaults to DEFAULT_STYLE.
            target_fps: Target frames per second for the game. Defaults to 60.
        """
        self.screen_width = APP_SCREEN_WIDTH
        self.screen_height = APP_SCREEN_HEIGHT
        self.title: bytes = title
        self.base_shader_path: Path = base_shader_path
        self.gui_style: bytes = gui_style
        self.config_flags: int = FLAG_VSYNC_HINT | FLAG_WINDOW_RESIZABLE
        self.time_loc: int = -1
        self.res_loc: int = -1
        self.mouse_loc: int = -1
        self.shader: Optional[Shader] = None
        self.texture_mapping: Dict[str, Texture2D] = defaultdict()
        self.fps: int = target_fps
        self.game_time: Optional[float] = None
        self.state_manager: StateManager = StateManager()

    def shutdown(self) -> None:
        """Cleanup and close the game window."""
        # Unload all textures
        for texture in self.texture_mapping.values():
            UnloadTexture(texture)

        # Unload shader
        if self.shader:
            UnloadShader(self.shader)

        CloseWindow()

    def initialize(self) -> None:
        """Initialize the game window, shaders, and resources.

        This method sets up the raylib window with the configured settings,
        loads and preprocesses the fragment shader, loads the GUI style,
        and loads all dice textures into the texture mapping.

        The shader is preprocessed to resolve any #include directives before
        being loaded into memory.
        """
        SetConfigFlags(self.config_flags)
        InitWindow(self.screen_width, self.screen_height, self.title)
        SetTargetFPS(self.fps)
        frag_src: str = self.preprocess_shader(self.base_shader_path.read_text(),
                                                self.base_shader_path.parent)
        self.shader = LoadShaderFromMemory(ffi.NULL, frag_src.encode())
        self.time_loc = GetShaderLocation(self.shader, b"u_time")
        self.res_loc = GetShaderLocation(self.shader, b"u_resolution")
        self.mouse_loc = GetShaderLocation(self.shader, b"u_mouse")
        GuiLoadStyle(self.gui_style)
        for texture_name, texture_path in DICE_TEXTURE_MAP.__dict__.items():
            self.texture_mapping[texture_name] = LoadTexture(texture_path)
        self.game_time = GetTime()


    def preprocess_shader(self, source: str, directory: Path) -> str:
        """Preprocess shader source code by resolving #include directives.

        This method recursively processes shader source code, replacing any
        #include "filename" directives with the contents of the included file.
        If an included file doesn't exist, the directive is replaced with an
        empty string.

        Args:
            source: The shader source code to preprocess.
            directory: The directory containing the shader file, used to resolve
                relative include paths.

        Returns:
            The preprocessed shader source code with all includes resolved.
        """
        pattern: re.Pattern = re.compile(r'#include\s+"([^"]+)"')

        def replace_include(match: re.Match) -> str:
            """Replace an include directive with the file contents.

            Args:
                match: Regex match object containing the include filename.

            Returns:
                The contents of the included file, or empty string if not found.
            """
            inc_path: Path = directory / match.group(1)
            if not inc_path.exists():
                return ""
            return self.preprocess_shader(inc_path.read_text(), inc_path.parent)

        return pattern.sub(replace_include, source)


    def setup_states(self) -> None:
        """Setup and register all game states with the state manager.

        This method should be called after initialize() to register all
        available game states (title, gameplay, etc.) and set the initial state.
        """
        from game.title_state import TitleState
        # Import other states as needed
        from game.game_state import GameStateDeploy
        from game.end_state import EndState

        # Register states
        self.state_manager.add_state("TITLE", TitleState(self))
        self.state_manager.add_state("GAME", GameStateDeploy(self))
        self.state_manager.add_state("END", EndState(self))

        # Set initial state
        self.state_manager.change_state("TITLE")

    def run_loop(self) -> None:
        """Run the main game loop.

        This method runs the game loop, updating and rendering the current
        state until the window is closed.
        """
        while not WindowShouldClose():
            delta_time = GetFrameTime()
            if delta_time > 0.05:
                delta_time = 0.05
            BeginDrawing()
            ClearBackground(RAYWHITE)

            # Update and render current state
            self.state_manager.update(delta_time)
            self.state_manager.render()

            EndDrawing()

    async def run_loop_async(self) -> None:
        """
        Run the game loop asynchronously.
        """
        while not WindowShouldClose():
            delta_time = GetFrameTime()
            if delta_time > 0.05:
                delta_time = 0.05
            BeginDrawing()
            ClearBackground(RAYWHITE)

            # Update and render current state
            self.state_manager.update(delta_time)
            self.state_manager.render()

            EndDrawing()
            await asyncio.sleep(0)

    def shutdown(self) -> None:
        """Cleanup and close the game window.

        This method should be called when the game is exiting to properly
        clean up resources.
        """
        CloseWindow()