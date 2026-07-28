from raylib import *
from abc import ABC
from game.base import Game

class Component:
    def __init__(self, name: str,game: Game):
        self.name = name
        self.enabled = True

    def update(self, delta_time: float):
        pass
