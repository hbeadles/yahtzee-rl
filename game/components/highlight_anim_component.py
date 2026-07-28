from game.components.component import Component
from game.base import Game
from raylib import Lerp

class HighlightAnimComponent(Component):
    def __init__(self,
                 name: str,
                 game: Game,
                 duration: float = 2.0,
                 transition_speed: float = 3.0):
        super().__init__(name, game)
        self.duration = duration
        self.transition_speed = transition_speed
        self._t = 0.2
        self._timer = 0.0
        self._highlighted = False

    
    def fire(self):
        self._highlighted = True
        self._timer = self.duration
        self._t = 0.2
        
    @property
    def intensity(self) -> float:
        return self._t

    def render_color(self, start_color: tuple[int, int, int], end_color: tuple[int, int, int]) -> tuple[int, int, int]:
        return tuple(int(Lerp(start_color[i], end_color[i], self._t)) for i in range(3))
    
    def update(self, delta_time: float):
        target = 1.0 if self._timer > self.duration * 0.5 else 0.0
        alpha = self.transition_speed * delta_time
        self._t = Lerp(self._t, target, alpha)
        if self._highlighted:
            self._timer -= delta_time
            if self._timer <= 0:
                self._highlighted = False
        # if self._highlighted:
        #     self._timer -= delta_time
        #     if self._timer <= 0:
        #         self._highlighted = False
        #
        #     target = 1.0 if self._timer > self.duration * 0.5 else 0.0
        #     self._t = Lerp(self._t, target, self.transition_speed * delta_time)
        # else:
    
    