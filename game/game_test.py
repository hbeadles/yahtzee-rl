from raylib import *
from game import APP_SCREEN_WIDTH, APP_SCREEN_HEIGHT
from yahtzee_rl.config import CATEGORY_NAMES
import re
from pathlib import Path

category_scores =[ False for _ in CATEGORY_NAMES]
category_names = [name.encode("utf-8") for name in CATEGORY_NAMES]
scroll_index = ffi.new("int *", 0)
active = ffi.new("int *", -1)
time_loc = -1
res_loc = -1
mouse_loc = -1
shader = None
dice_texture = None
dice_texture_normal = None
dice_texture_hover = None
selection_cursor = None
game_time = None
dice_animate_time = None
dice_source_x, dice_source_y = 0, 0
def preprocess_shader(source, directory):
    pattern = re.compile(r'#include\s+"([^"]+)"')
    def replace_include(match):
        inc_path = directory / match.group(1)
        if not inc_path.exists():
            return ""
        return preprocess_shader(inc_path.read_text(), inc_path.parent)
    return pattern.sub(replace_include, source)

def blitFromSpriteSheet(texture, row, column, pixel_width, pixel_height, dest_rect):
    source_x = column * pixel_width
    source_y = row * pixel_height
    source_rect = ffi.new("struct Rectangle *", [source_x, source_y, pixel_width, pixel_height])
    DrawTexturePro(
        texture,
        source_rect[0],
        dest_rect[0],
        ffi.new("struct Vector2 *", [0.0, 0.0])[0],
        0.0,
        WHITE
    )

def initialize(frag_path: Path):
    global shader, time_loc, res_loc, mouse_loc, dice_texture, \
        game_time, dice_texture_normal, dice_texture_hover, selection_cursor
    SetConfigFlags(FLAG_VSYNC_HINT | FLAG_WINDOW_RESIZABLE)
    InitWindow(APP_SCREEN_WIDTH, APP_SCREEN_HEIGHT, b"Yahtzee RL")
    SetTargetFPS(60)
    frag_src = preprocess_shader(frag_path.read_text(), frag_path.parent)
    shader = LoadShaderFromMemory(ffi.NULL, frag_src.encode())
    time_loc = GetShaderLocation(shader, b"u_time")
    res_loc = GetShaderLocation(shader, b"u_resolution")
    mouse_loc = GetShaderLocation(shader, b"u_mouse")
    GuiLoadStyle(b"style_rltech.rgs")
    dice_texture = LoadTexture(b"textures/six_sided_die.png")
    dice_texture_normal = LoadTexture(b"textures/dice_full_normal.png")
    dice_texture_hover = LoadTexture(b"textures/dice_full_hover.png")
    selection_cursor = LoadTexture(b"textures/selection_cursor.png")
    game_time = GetTime()
def app():
    initialize(Path("shaders/lines.frag"))
    while not WindowShouldClose():
        result = process_input()
        if not result:
            break
        update_game()
        generate_output()
    shutdown()

def process_input():
    if IsKeyPressed(KEY_ESCAPE):
        return False
    else:
        return True

def animate_dice(cur_time):
    global dice_animate_time
    if not dice_animate_time:
        dice_animate_time = GetTime()
    elapsed = cur_time - dice_animate_time
    row = 14
    indexes = 5
    anim_fps = 4.0
    frame_count = int(elapsed * anim_fps)
    current_frame = frame_count % 6
    if current_frame == 6:
        dice_animate_time = GetTime()
    source_x = current_frame * 16
    source_y = 14 * 16

    return (source_x, source_y)


def update_game():
    global game_time, dice_source_x, dice_source_y
    game_time = GetTime()
    if IsShaderValid(shader):
        t = ffi.new("float *", GetTime())
        SetShaderValue(shader, time_loc, t, SHADER_UNIFORM_FLOAT)

        res = ffi.new("float[2]", [float(GetScreenWidth()), float(GetScreenHeight())])
        SetShaderValue(shader, res_loc, res, SHADER_UNIFORM_VEC2)

        mp = GetMousePosition()
        mouse = ffi.new("float[2]", [mp.x, float(GetScreenHeight()) - mp.y])
        SetShaderValue(shader, mouse_loc, mouse, SHADER_UNIFORM_VEC2)

    dice_source_x, dice_source_y = animate_dice(game_time)
def generate_output():
    BeginDrawing()
    ClearBackground(RAYWHITE)

    # Layer 1
    if IsShaderValid(shader):
        BeginShaderMode(shader)
        DrawRectangle(0, 0, APP_SCREEN_WIDTH, APP_SCREEN_HEIGHT, WHITE)
        EndShaderMode()

    GuiPanel(ffi.new("struct Rectangle *", [0, 0, 260, APP_SCREEN_HEIGHT])[0], b"Scorecard")
    GuiGroupBox(ffi.new("struct Rectangle *", [10, 40, 240, 200])[0], b"Upper Section")

    y = 55
    for i in range(6):
        rect = ffi.new("struct Rectangle *", [20, y, 220, 25])[0]
        if category_scores[i]:
            GuiDisable()
        if GuiButton(rect, category_names[i]):
            category_scores[i] = not category_scores[i]
        GuiEnable()
        y += 30
    GuiGroupBox(ffi.new("struct Rectangle *", [10, 260, 240, 220])[0], b"Lower Section")
    y = 275
    for i in range(6, 13):
        rect = ffi.new("struct Rectangle *", [20, y, 220, 25])[0]
        if category_scores[i]:
            GuiDisable()
        if GuiButton(rect, category_names[i]):
            category_scores[i] = not category_scores[i]
        GuiEnable()
        y += 28

    DrawText(b"Yahtzee", APP_SCREEN_WIDTH // 2, 20, 24, WHITE)
    
    test_source = ffi.new("struct Rectangle *", [dice_source_x, dice_source_y, 16, 16])
    test_dests_y = APP_SCREEN_HEIGHT // 2 - 175
    test_dests_x = APP_SCREEN_WIDTH // 2 - 75


    origin = ffi.new("struct Vector2 *", [0.0, 0.0])[0]
    GuiSetAlpha(0.75)
    GuiPanel(ffi.new("struct Rectangle *", [APP_SCREEN_WIDTH // 2 - 85,
                                                  APP_SCREEN_HEIGHT // 2  - 200,
                                                  400, 100])[0], b"Dice Roll")
    GuiSetAlpha(1.0)
    for x in range(5):
        test_rect = ffi.new("struct Rectangle *", [test_dests_x, test_dests_y, 64, 64])
        DrawTexturePro(dice_texture,
                       test_source[0],
                       test_rect[0], origin, 0.0, WHITE)
        test_dests_x += 75
    GuiSetAlpha(0.75)
    GuiPanel(ffi.new("struct Rectangle *", [APP_SCREEN_WIDTH // 2 - 85,
                                                  APP_SCREEN_HEIGHT // 2  + 100,
                                                  475, 100])[0], b"Dice Hover Effect")
    test_hover_x = APP_SCREEN_WIDTH // 2 - 75
    test_hover_y = APP_SCREEN_HEIGHT // 2 + 130
    mouse_pos = GetMousePosition()
    for x in [1, 2, 3, 4, 5, 6]:
        test_rect = ffi.new("struct Rectangle *", [test_hover_x, test_hover_y, 64, 64])
        blitFromSpriteSheet(dice_texture_hover, 0, x, 22, 22, test_rect)
        is_hovering = CheckCollisionPointRec(mouse_pos, test_rect[0])
        if is_hovering:
            cursor_dest_rect = ffi.new("struct Rectangle *", [test_hover_x - 5, test_hover_y - 5, 74, 74])
            blitFromSpriteSheet(selection_cursor, 0, 0, 32, 32, cursor_dest_rect)
        test_hover_x += 75

    GuiSetAlpha(1.0)
    DrawFPS(10, 10)
    EndDrawing()

def shutdown():
    CloseWindow()

if __name__ == "__main__":
    app()