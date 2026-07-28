from raylib import *
from game.objects.game_object import GameObject
from game.components.dice_anim_component import DiceAnimPhaseComponent, PhaseConfig
from game.components.highlight_anim_component import HighlightAnimComponent
from game.base import Game
from game import DiceGameStates 
from game.drawing import blitFromSpriteSheet
from game import YahtzeeGameStates


class DiceRollObject(GameObject):
    def __init__(self, name, state, game, num_dice=5):
        super().__init__(name, state, game)
        self.num_dice = num_dice
        self.selected: list[bool] = [False for _ in range(num_dice)]
        self.roll_requested = False
        self.hover_texture = self.game.texture_mapping["cursor_hover"]
        phases = {
            DiceGameStates.DICE_IDLE: PhaseConfig(
                texture_key="normal_path", frame_w=22, frame_h=22,
                row=0, num_frames=7, anim_fps=2.0, loop=False,
            ),
            DiceGameStates.DICE_ROLL: PhaseConfig(
                texture_key="rolling_path", frame_w=16, frame_h=16,
                row=14, num_frames=7, anim_fps=4.0, loop=True,
            ),
            DiceGameStates.DICE_READY: PhaseConfig(
                texture_key="normal_path", frame_w=22, frame_h=22,
                row=0, num_frames=7, anim_fps=2.0, loop=False,
            ),
            DiceGameStates.DICE_LOCKED: PhaseConfig(
                texture_key="normal_path", frame_w=22, frame_h=22,
                row=0, num_frames=7, anim_fps=2.0, loop=False,
            ),
        }
        for i in range(num_dice):
            self.add(DiceAnimPhaseComponent(
                name=f"die_{i}",
                game=game,
                phases=phases,
                initial_phase=DiceGameStates.DICE_IDLE,
            ))
            self.add(HighlightAnimComponent(
                name=f"die_highlight_{i}",
                game=game,
                duration=0.5,
                transition_speed=3.0,
            ))

    def start_roll(self, dice_values: list[int], stagger: float = 0.4):
        for i in range(self.num_dice):
            comp = self.get(f"die_{i}")
            comp.set_value(dice_values[i])
            if self.selected[i]:
                comp.current_phase = DiceGameStates.DICE_IDLE
            comp.reset(start_delay=stagger * (i + 1))

    def toggle_select(self, index: int):
        self.selected[index] = not self.selected[index]


    def selected_mask(self) -> int:
        mask = 0
        for i, reroll in enumerate(self.selected):
            if reroll:
                mask |= (1 << i)
        return mask

    def clear_selection(self):
        self.selected = [False] * self.num_dice

    def all_ready(self) -> bool:
        return all(
            self.get(f"die_{i}").current_phase == DiceGameStates.DICE_READY
            for i in range(self.num_dice)
        )

    def highlight_dice(self, indices: list[int]):
        """Trigger a brief highlight pulse on the specified dice."""
        for i in indices:
            comp = self.get(f"die_highlight_{i}")
            if comp:
                comp.fire()

    def render(self):
        GuiSetAlpha(0.75)
        GuiPanel(ffi.new("struct Rectangle *", [self.game.screen_width // 2 - 85,
                                                self.game.screen_height // 2 - 200,
                                                400, 100])[0], b"Dice Roll")

        if GuiButton(ffi.new("struct Rectangle *", [self.game.screen_width // 2 + 150,
                                                    self.game.screen_height // 2 - 100,
                                                    85, 30])[0], b"Reroll Dice"):
            self.roll_requested = True

        GuiSetAlpha(1.0)
        base_x = self.game.screen_width // 2 - 75
        base_y = self.game.screen_height // 2 - 175
        for i in range(self.num_dice):
            comp = self.get(f"die_{i}")
            highlight_comp = self.get(f"die_highlight_{i}")
            cfg = comp.current_config
            texture = self.game.texture_mapping[cfg.texture_key]
            source_rect = ffi.new("struct Rectangle *", [
                comp.source_x, comp.source_y, cfg.frame_w, cfg.frame_h
            ])
            dest_rect = ffi.new("struct Rectangle *", [
                base_x + i * 75, base_y, 64, 64
            ])
            if highlight_comp and highlight_comp._highlighted:
                tint = highlight_comp.render_color((255, 128, 128), (140, 0, 0, 255))
                DrawTexturePro(texture, source_rect[0], dest_rect[0],
                           ffi.new("struct Vector2 *", [0.0, 0.0])[0], 0.0, (tint[0], tint[1], tint[2], 255))
            elif self.selected[i]:
                # Tint to indicate die is marked for reroll
                DrawTexturePro(texture, source_rect[0], dest_rect[0],
                           ffi.new("struct Vector2 *", [0.0, 0.0])[0], 0.0, LIGHTGRAY)
            else:
                DrawTexturePro(texture, source_rect[0], dest_rect[0],
                           ffi.new("struct Vector2 *", [0.0, 0.0])[0], 0.0, WHITE)
            mouse_pos = GetMousePosition()
            is_hovering = CheckCollisionPointRec(mouse_pos, dest_rect[0])
            if comp.current_phase == DiceGameStates.DICE_READY and is_hovering and \
            (self.state.yahtzee_game_state == YahtzeeGameStates.GAME_DICE_SELECTION_ACTION):
                # Draw a highlight around the dice
                cursor_dest_rect = ffi.new("struct Rectangle *", [base_x + i * 75 - 5, base_y - 5, 74, 74])
                blitFromSpriteSheet(self.hover_texture, 0, 0, 32, 32, cursor_dest_rect)
                if IsMouseButtonPressed(MOUSE_BUTTON_LEFT):
                    self.toggle_select(i)
                    



        

