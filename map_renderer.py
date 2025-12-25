import io
import time
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw


def render_minimap_png(
    points_xy_m: List[Tuple[float, float]],
    size: int = 600,
    meters_range: float = 6.0,
    rotate_deg: float = 0.0,
    stamp: Optional[str] = None
) -> bytes:
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx = cy = size // 2
    px_per_m = (size / 2) / meters_range

    # grid
    for m in range(1, int(meters_range) + 1):
        r = int(m * px_per_m)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(35, 35, 35))

    draw.line((0, cy, size, cy), fill=(50, 50, 50))
    draw.line((cx, 0, cx, size), fill=(50, 50, 50))
    draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(255, 255, 255))

    if rotate_deg != 0:
        th = np.deg2rad(rotate_deg)
        c, s = np.cos(th), np.sin(th)
    else:
        c, s = 1.0, 0.0

    for x, y in points_xy_m:
        xr = x * c - y * s
        yr = x * s + y * c
        px = int(cx + xr * px_per_m)
        py = int(cy - yr * px_per_m)
        if 0 <= px < size and 0 <= py < size:
            img.putpixel((px, py), (0, 255, 0))

    if stamp is None:
        stamp = time.strftime("%H:%M:%S")
    draw.text((10, 10), f"points={len(points_xy_m)} t={stamp}", fill=(200, 200, 200))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
