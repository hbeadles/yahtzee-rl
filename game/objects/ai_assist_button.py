from game.objects.gui_button_texture_object import GuiButtonTextureObject
from game.state import State
from game.base import Game
from yahtzee_rl.strategies.markov import MarkovStrategy
import numpy as np

class AiAssistButton(GuiButtonTextureObject):
    def __init__(self,
                 name: str,
                 button_text: bytes,
                 width: int,
                 height: int,
                 state: State,
                 game: Game):
        super().__init__(name, button_text, width, height, state, game)
        self.markov_strategy = MarkovStrategy(state.yahtzee_env)

    def on_click(self):
        parsed_obs = self.state.parsed_observation
        action = self.markov_strategy.strategy_dict(parsed_obs)
        rolls_left = parsed_obs['rolls_remaining']
        if rolls_left > 0:
            roll_bits = np.unpackbits(np.array([int(action)], dtype=np.uint8), count=5, bitorder='little')
            reroll = np.argwhere(roll_bits == 1)
            self.state.guide_text.show_message(f"AI would roll dice at positions: {reroll.tolist()}")
            self.state.dice_row.highlight_dice(reroll.flatten().tolist())

        else:
            self.state.guide_text.show_message(f"AI would choose to score in category: {action}")
        
        self.pressed = False