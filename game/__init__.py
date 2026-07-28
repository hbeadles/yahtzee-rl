from enum import Enum
from dataclasses import dataclass
APP_SCREEN_WIDTH = 1024
APP_SCREEN_HEIGHT = 768

@dataclass
class DiceTextureMap:
    rolling_path: bytes
    normal_path: bytes
    hover_path: bytes
    cursor_hover: bytes


DICE_TEXTURE_MAP = DiceTextureMap(
    b"game/textures/six_sided_die.png",
    b"game/textures/dice_full_normal.png",
    b"game/textures/dice_full_hover.png",
    b"game/textures/selection_cursor.png"
)

DEFAULT_STYLE = b"game/style_rltech.rgs"

class YahtzeeGameStates(Enum):
    GAME_START = 1,
    GAME_DICE_ROLLING_ACTION = 2,
    GAME_DICE_SELECTION_ACTION = 3,
    GAME_SCORECARD_SELECTION_ACTION = 4,
    GAME_ENDED = 5,
    GAME_ROUND_END = 6

class DiceGameStates(Enum):
    DICE_IDLE = 1,
    DICE_HOVER = 2,
    DICE_ROLL = 3,
    DICE_READY = 4
    DICE_LOCKED = 5

@dataclass
class DiceAnimationState:
    phase: DiceGameStates = DiceGameStates.DICE_IDLE,
    elapsed_time: float= 0.0,
    value: int = 0
    start_delay: float = 0.0,
    source_x: int = 0,
    source_y: int = 0

class ScorecardGameStates(Enum):
    SCORECARD_IDLE = 1,
    SCORECARD_LOCKED = 2

@dataclass
class SpriteSource:
    texture_key: str      # key into game.texture_mapping
    frame_w: int
    frame_h: int
    row: int              # fixed row in sheet
    min_col: int
    max_col: int

SPRITE_ROLLING = SpriteSource('rolling_path', 16, 16, row=14, min_col=0, max_col=6)
SPRITE_NORMAL = SpriteSource('normal_path', 22, 22, row=0, min_col=1, max_col=7)

PHASE_SPRITE: dict[DiceGameStates, SpriteSource] = {
    DiceGameStates.DICE_IDLE: SPRITE_NORMAL,
    DiceGameStates.DICE_ROLL: SPRITE_ROLLING,
    DiceGameStates.DICE_READY: SPRITE_NORMAL,
}
