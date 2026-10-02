"""Pre-render every (pose, tint) hand as a trimmed PNG sprite for the web page.

The page must stay a single self-contained file that opens from disk, so it cannot run
three.js or load a GLB at view time. Eight poses times three tints is twenty-four
images -- small enough to inline, and it means the page shows exactly the hands the GIF
shows rather than a second, divergent drawing of them.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from PIL import Image

from render_hands import BAD, GOOD, NEUTRAL, ORDER, POSES, HandRenderer, static_server

TINTS = {"neutral": NEUTRAL, "good": GOOD, "bad": BAD}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("assets/render/sprites"))
    ap.add_argument("--max-width", type=int, default=210)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # One crop box for every sprite, from the union of all poses, so the hand does not
    # jump about when the pose changes.
    with static_server(Path("assets")) as origin:
        r = HandRenderer(f"{origin}/render/hand.html")
        try:
            r.frame_all([POSES[k] for k in ORDER])
            boxes = []
            raw = {}
            for key in ORDER:
                im = Image.open(io.BytesIO(r.render(POSES[key], NEUTRAL)))
                raw[key] = im
                if im.getbbox():
                    boxes.append(im.getbbox())
            L = min(b[0] for b in boxes); T = min(b[1] for b in boxes)
            R = max(b[2] for b in boxes); B = max(b[3] for b in boxes)
            pad = 8
            box = (max(L - pad, 0), max(T - pad, 0), R + pad, B + pad)
            print(f"shared crop {box}  ->  {box[2]-box[0]}x{box[3]-box[1]}")

            total = 0
            for key in ORDER:
                for tname, tint in TINTS.items():
                    im = Image.open(io.BytesIO(r.render(POSES[key], tint))).crop(box)
                    if im.width > args.max_width:
                        h = round(im.height * args.max_width / im.width)
                        im = im.resize((args.max_width, h), Image.LANCZOS)
                    # Quantise: these are smooth matte renders on transparency, so a
                    # palette costs nothing visible and roughly quarters the bytes that
                    # end up base64-encoded inside the page.
                    q = im.quantize(colors=160, method=Image.FASTOCTREE)
                    p = args.out / f"{key}_{tname}.png"
                    q.save(p, optimize=True)
                    total += p.stat().st_size
            print(f"{len(ORDER) * len(TINTS)} sprites, {total/1024:.0f} KB total")
        finally:
            r.close()


if __name__ == "__main__":
    main()
