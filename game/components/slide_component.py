from game.components.component import Component
from game.base import Game


class SlideComponent(Component):
    """Reusable slide animation that lerps between an open and closed position."""

    def __init__(self,
                 name: str,
                 game: Game,
                 closed_pos: float,
                 open_pos: float,
                 speed: float = 8.0):
        super().__init__(name, game)
        self.closed_pos = closed_pos
        self.open_pos = open_pos
        self.speed = speed
        self.is_open = False
        self.current_pos = closed_pos

    def toggle(self):
        self.is_open = not self.is_open

    @property
    def target_pos(self) -> float:
        return self.open_pos if self.is_open else self.closed_pos

    @property
    def is_settled(self) -> bool:
        return abs(self.current_pos - self.target_pos) < 0.5

    def update(self, delta_time: float):
        target = self.target_pos
        self.current_pos += (target - self.current_pos) * min(1.0, delta_time * self.speed)
        if self.is_settled:
            self.current_pos = target
