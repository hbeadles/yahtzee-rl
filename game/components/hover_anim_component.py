from game.components.component import Component
from game.base import Game
from raylib import Lerp
import math

class HoverAnimComponent(Component):
    def __init__(self, 
                 name: str, 
                 game: Game,
                 max_scale: float = 1.08,
                 transition_speed: float = 2.0,
                 max_rotation: float = 10.0):
        super().__init__(name, game)
        self.max_scale = max_scale
        self.transition_speed = transition_speed
        self.max_rotation = max_rotation
        self._t = 0.0
        self._elapsed = 0.0
        self.hovering: bool = False
    @property
    def rotate(self) -> float:
        eased = self._ease_cubic(self._t)
        return self.max_rotation * math.sin(self._elapsed * 3.0) * eased
    @property
    def scale(self) -> float:
        eased = self._ease_cubic(self._t)
        return 1.0 + (self.max_scale - 1.0) * eased
    
    def update(self, delta_time: float):
        target = 1.0 if self.hovering else 0.0
        alpha = self.transition_speed * delta_time
        self._t = Lerp(self._t, target, alpha)
        if self.hovering:
            self._elapsed += delta_time
        else:
            self._elapsed = 0.0

    @staticmethod
    def _ease_cubic(t: float) -> float:
        if t < 0.5:
            return 4.0 * t * t * t
        return 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0