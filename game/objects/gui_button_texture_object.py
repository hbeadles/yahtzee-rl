from game.objects.game_object import GameObject
from game.components.hover_anim_component import HoverAnimComponent
from raylib import *
from game.state import State
from game.base import Game
import math
from abc import abstractmethod

class GuiButtonTextureObject(GameObject):

    def __init__(self,
                 name: str,
                 button_text: bytes,
                 width: int,
                 height: int,
                 state: State,
                 game: Game):
        super().__init__(name, state, game)
        self.add(HoverAnimComponent(
            name="hover_anim",
            game=game,
            max_scale=1.18,
            transition_speed=3.0,
            max_rotation=10.0
        ))
        self.dest_x = self.game.screen_width // 2 - 85
        self.dest_y = self.game.screen_height // 2 + 60
        self.pressed = False
        self.btn_w = width
        self.btn_h = height
        self.max_scale = 1.1
        self.button_text = button_text
        self.render_texture = LoadRenderTexture(self.btn_w, self.btn_h)
        self.btn_rect_local = ffi.new("struct Rectangle *", [0, 0, self.btn_w, self.btn_h])
        self.scale = 1.0
        self.hit_rect = ffi.new("struct Rectangle *", [
            self.dest_x, self.dest_y, self.btn_w, self.btn_h
        ])
        self.dest_rect = ffi.new("struct Rectangle *", [
            self.dest_x + self.btn_w / 2,
            self.dest_y + self.btn_h / 2,
            self.btn_w, self.btn_h
        ])
        self.source_rect = ffi.new("struct Rectangle *", [0, 0, self.btn_w, -self.btn_h])
        self.rotation = 0.0

    def update(self, delta_time: float):
        hover_anim = self.get("hover_anim")
        if hover_anim:
            hover_anim.update(delta_time)

        mouse = GetMousePosition()
        if hover_anim:
            hover_anim.hovering = CheckCollisionPointRec(mouse, self.hit_rect[0]) 

        if hover_anim and hover_anim.hovering and IsMouseButtonPressed(MOUSE_BUTTON_LEFT):
            self.pressed = True
            self.on_click()
        super().update(delta_time)

    @abstractmethod
    def on_click(self):
        pass

    def render(self):
        hover_anim = self.get("hover_anim")
        
        BeginTextureMode(self.render_texture)
        ClearBackground(BLANK)
        GuiButton(
            self.btn_rect_local[0],
            self.button_text
        )
        EndTextureMode()
        if hover_anim:
            scale = hover_anim.scale
        else:
            scale = 1.0
        if hover_anim:
            rotation = hover_anim.rotate
        else:
            rotation = 0.0
        dest_w = self.btn_w * scale
        dest_h = self.btn_h * scale
        dest_rect = ffi.new("struct Rectangle *", [
            self.dest_x + self.btn_w / 2,
            self.dest_y + self.btn_h / 2,
            dest_w, dest_h
        ])
        DrawTexturePro(
            self.render_texture.texture,
            self.source_rect[0],
            dest_rect[0],
            ffi.new("struct Vector2 *", [dest_w / 2, dest_h / 2])[0],
            rotation,
            WHITE
        )