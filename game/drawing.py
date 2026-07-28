from raylib import DrawTexturePro, ffi, WHITE

def blitFromSpriteSheet(texture,
                        row,
                        column,
                        pixel_width,
                        pixel_height,
                        dest_rect):
    """Given a provided texture, render a specific
    sprite image
    Args:
        texture: The texture to render from
        row: The row of the sprite to render
        column: The column of the sprite to render
        pixel_width: The width of a single sprite
        pixel_height: The height of a single sprite
        dest_rect: The rectangle to render to
    Returns:
        None
    """
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

