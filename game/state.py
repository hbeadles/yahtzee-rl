"""State management module for the Yahtzee RL game.

This module provides the base State class and StateManager for managing
different game screens and transitions between them (e.g., title screen,
gameplay screen, game over screen).
"""

from typing import Dict, Optional, TYPE_CHECKING, Any
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from game.base import Game

class Message:
    def __init__(self, message_type: str, message_data: Any):
        self.message_type = message_type
        self.message_data = message_data

    def get_message_type(self) -> str:
        return self.message_type

    def get_message_data(self) -> Any:
        return self.message_data

class State(ABC):
    """Abstract base class for all game states/screens.

    Each state represents a distinct screen or mode in the game (e.g., title
    screen, gameplay, pause menu). States handle their own input processing,
    update logic, and rendering.

    Attributes:
        game: Reference to the main Game instance for accessing shared resources.
    """

    def __init__(self, game: 'Game') -> None:
        """Initialize the state.

        Args:
            game: Reference to the main Game instance.
        """
        self.game: 'Game' = game

    @abstractmethod
    def update(self, delta_time: float) -> Optional[str]:
        """Update the state logic.

        This method is called every frame to update the state's logic,
        animations, and handle input processing.

        Args:
            delta_time: Time elapsed since the last frame in seconds.

        Returns:
            The name of the next state to transition to, or None to stay
            in the current state.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self) -> None:
        """Render the state's visuals.

        This method is called every frame to draw the state's UI and graphics.
        It should be called between BeginDrawing() and EndDrawing().
        """
        raise NotImplementedError

    def on_enter(self) -> None:
        """Called when transitioning into this state.

        Override this method to perform initialization when entering the state,
        such as loading resources, resetting variables, or playing sounds.
        """
        pass

    def on_exit(self) -> None:
        """Called when transitioning out of this state.

        Override this method to perform cleanup when leaving the state,
        such as unloading resources or saving state.
        """
        pass

    def on_pause(self) -> None:
        """Called when this state is paused (for stack-based state management).

        Override this method to handle pausing, such as when a pause menu
        is pushed on top of the gameplay state.
        """
        pass

    def on_resume(self) -> None:
        """Called when this state is resumed (for stack-based state management).

        Override this method to handle resuming from a paused state.
        """
        pass


class StateManager:
    """Manages game states and transitions between them.

    The StateManager maintains a collection of states and handles transitions
    between them. It supports both simple state switching and stack-based
    state management for features like pause menus.

    Attributes:
        states: Dictionary mapping state names to State instances.
        current_state: The currently active state, or None if no state is active.
        state_stack: Stack of states for push/pop operations.
    """

    def __init__(self) -> None:
        """Initialize the StateManager with no states."""
        self.states: Dict[str, State] = {}
        self.current_state: Optional[State] = None
        self.state_stack: list[State] = []
        self.state_messager: dict[State, Message] = {

        }

    def add_state(self, name: str, state: State) -> None:
        """Register a state with the manager.

        Args:
            name: Unique identifier for the state (e.g., "TITLE", "GAME", "PAUSE").
            state: The State instance to register.

        Raises:
            ValueError: If a state with the given name already exists.
        """
        if name in self.states:
            raise ValueError(f"State '{name}' already exists in StateManager")
        self.states[name] = state

    def change_state(self, name: str) -> None:
        """Transition to a different state.

        This method exits the current state (if any), clears the state stack,
        and enters the new state.

        Args:
            name: The name of the state to transition to.

        Raises:
            KeyError: If no state with the given name exists.
        """
        if name not in self.states:
            raise KeyError(f"State '{name}' not found in StateManager")

        # Exit current state
        if self.current_state:
            self.current_state.on_exit()

        # Clear the stack when changing states
        self.state_stack.clear()

        # Enter new state
        self.current_state = self.states[name]
        self.current_state.on_enter()

    def push_state(self, name: str) -> None:
        """Push a new state onto the stack.

        This is useful for overlay states like pause menus. The current state
        is paused and the new state becomes active.

        Args:
            name: The name of the state to push.

        Raises:
            KeyError: If no state with the given name exists.
        """
        if name not in self.states:
            raise KeyError(f"State '{name}' not found in StateManager")

        # Pause current state if it exists
        if self.current_state:
            self.current_state.on_pause()
            self.state_stack.append(self.current_state)

        # Enter new state
        self.current_state = self.states[name]
        self.current_state.on_enter()

    def pop_state(self) -> None:
        """Pop the current state and return to the previous one.

        This exits the current state and resumes the state that was on top
        of the stack.

        Raises:
            RuntimeError: If the state stack is empty.
        """
        if not self.state_stack:
            raise RuntimeError("Cannot pop state: state stack is empty")

        # Exit current state
        if self.current_state:
            self.current_state.on_exit()

        # Resume previous state
        self.current_state = self.state_stack.pop()
        self.current_state.on_resume()

    def update(self, delta_time: float) -> None:
        """Update the current state and handle state transitions.

        This method should be called every frame. It updates the current state
        and automatically handles state transitions if the state requests one.

        Args:
            delta_time: Time elapsed since the last frame in seconds.
        """
        if self.current_state:
            next_state = self.current_state.update(delta_time)
            if next_state:
                self.change_state(next_state)

    def render(self) -> None:
        """Render the current state.

        This method should be called every frame between BeginDrawing() and
        EndDrawing().
        """
        if self.current_state:
            self.current_state.render()

    def get_current_state_name(self) -> Optional[str]:
        """Get the name of the current state.

        Returns:
            The name of the current state, or None if no state is active.
        """
        if not self.current_state:
            return None

        for name, state in self.states.items():
            if state is self.current_state:
                return name
        return None

    def has_state(self, name: str) -> bool:
        """Check if a state with the given name exists.

        Args:
            name: The state name to check.

        Returns:
            True if the state exists, False otherwise.
        """
        return name in self.states


