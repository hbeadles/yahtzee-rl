from raylib import *
from game.components.component import Component
from typing import Any, Dict
from game.base import Game
from game import DiceGameStates
from dataclasses import dataclass

@dataclass
class PhaseConfig:
    """All sprite sheet info needed to render one phase."""
    texture_key: str
    frame_w: int
    frame_h: int
    row: int
    num_frames: int       # how many frames in this phase
    anim_fps: float
    loop: bool = True     # False → hold last frame, then fire on_complete


class DiceAnimPhaseComponent(Component):
    """Sprite animation that can transition through multiple phases,
    each with its own texture key and sprite sheet region.

    Example: DICE_ROLL (rolling spritesheet) → DICE_READY (faces spritesheet)
    """

    def __init__(self,
                 name: str,
                 game: Game,
                 phases: Dict[DiceGameStates, PhaseConfig],
                 initial_phase: DiceGameStates):
        super().__init__(name, game)
        self.phases = phases
        self.current_phase = initial_phase
        self.elapsed_time: float = 0.0
        self.start_delay: float = 0.0
        self.value: int = 0
        self.source_x: int = 0
        self.source_y: int = 0
        self.on_phase_complete = None   # callback(phase) when a non-looping phase ends

    @property
    def current_config(self) -> PhaseConfig:
        return self.phases[self.current_phase]

    def set_value(self, value: int):
        self.value = value

    def reset(self, start_delay: float = 0.0):
        self.current_phase = DiceGameStates.DICE_IDLE
        self.elapsed_time = 0.0
        self.start_delay = start_delay

    def update(self, delta_time: float):
        self.elapsed_time += delta_time
        cfg = self.phases[self.current_phase]
        num_frames = cfg.num_frames
    
        if self.current_phase == DiceGameStates.DICE_IDLE:
            current_frame = 0
            if self.elapsed_time >= self.start_delay:
                self.current_phase = DiceGameStates.DICE_ROLL
                self.elapsed_time = 0.0
        elif self.current_phase == DiceGameStates.DICE_ROLL:
            frame_count = int(self.elapsed_time * cfg.anim_fps)
            current_frame = frame_count % num_frames  # 6 frames in the animation
            if frame_count >= num_frames:
                self.current_phase = DiceGameStates.DICE_READY
        elif self.current_phase == DiceGameStates.DICE_READY:
            current_frame = self.value

        else:
            current_frame = 0

        self.source_x = current_frame * cfg.frame_w
        self.source_y = cfg.row * cfg.frame_h
