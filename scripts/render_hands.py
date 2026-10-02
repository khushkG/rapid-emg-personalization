"""Render the eight hand poses with three.js in headless Chromium.

The 2D matplotlib hand it replaces could not be made to read convincingly: a closed
hand lost its fingers behind the palm, and poses that differ anatomically (a pinch and
a fist) are the same silhouette from one viewpoint. A real rigged mesh with lighting
solves both, because occlusion and shading do the work a flat drawing cannot.

Model: WebXR Input Profiles `generic-hand` (left), 25 standard XR hand joints, from
the W3C Immersive Web Working Group, under the W3C Software and Document License.
Renderer: three.js r160, MIT. Both are vendored under assets/ with their notices.

Poses are *deltas from the model's own bind pose*, so nothing depends on guessing its
rest rotations. Frames are written to disk one at a time and the browser page is
recycled between batches -- this host has stalled on memory repeatedly, and a long
lived page accumulating 240 framebuffers is exactly the shape of workload that did it.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import functools
import http.server
import re
import socketserver
import threading
import time
from pathlib import Path

# Eight poses as joint-angle sets. Chosen for mutual distinctness: a counting
# progression the eye reads instantly, plus four whole-hand shapes. The id-to-grasp
# mapping is unverified (no source states it), so legibility is the only criterion
# that can honestly be applied.
POSES: dict[str, dict] = {
    "rest": dict(curl=[0.28, 0.30, 0.32, 0.34], spread=0.35, thumb_abd=0.40,
                 thumb_curl=0.25, thumb_opp=0.20),
    "A":    dict(curl=[0.98, 1.00, 1.00, 0.98], spread=0.05, thumb_abd=0.15,
                 thumb_curl=0.85, thumb_opp=0.80),      # fist
    "B":    dict(curl=[0.02, 0.02, 0.02, 0.02], spread=1.00, thumb_abd=1.00,
                 thumb_curl=0.05, thumb_opp=0.00),      # all five, fanned
    "C":    dict(curl=[0.02, 1.00, 1.00, 1.00], spread=0.15, thumb_abd=0.15,
                 thumb_curl=0.85, thumb_opp=0.75),      # one finger
    "D":    dict(curl=[0.02, 0.02, 1.00, 1.00], spread=0.55, thumb_abd=0.15,
                 thumb_curl=0.85, thumb_opp=0.75),      # two fingers
    "E":    dict(curl=[0.02, 0.02, 0.02, 1.00], spread=0.42, thumb_abd=0.15,
                 thumb_curl=0.85, thumb_opp=0.75),      # three fingers
    "F":    dict(curl=[0.72, 0.80, 0.80, 0.72], spread=0.20, thumb_abd=1.00,
                 thumb_curl=0.05, thumb_opp=0.00),      # hook, thumb out
    "G":    dict(curl=[0.04, 0.04, 0.04, 0.04], spread=0.02, thumb_abd=0.02,
                 thumb_curl=0.45, thumb_opp=0.95),      # flat, thumb across
}
ORDER = ["rest", "A", "B", "C", "D", "E", "F", "G"]

NEUTRAL = 0xD3D6DB
GOOD = 0x9FD2B6
BAD = 0xF0ADA3


def lerp_pose(a: dict, b: dict, t: float) -> dict:
    t = max(0.0, min(1.0, t))
    out = {"curl": [x + (y - x) * t for x, y in zip(a["curl"], b["curl"])]}
    for k in ("spread", "thumb_abd", "thumb_curl", "thumb_opp"):
        out[k] = a[k] + (b[k] - a[k]) * t
    return out


@contextlib.contextmanager
def static_server(root: Path):
    """Serve `root` on a loopback port.

    Chromium refuses ES modules over file:// -- "Cross origin requests are only
    supported for protocol schemes: chrome, chrome-untrusted, data, http, https" --
    so the renderer needs a real origin. A thread-local http.server on an ephemeral
    port is the whole requirement; nothing leaves the machine.
    """
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(root.resolve()))

    class Quiet(socketserver.TCPServer):
        allow_reuse_address = True

        def handle_error(self, request, client_address):   # keep the log clean
            pass

    srv = Quiet(("127.0.0.1", 0), handler)
    srv.RequestHandlerClass.log_message = lambda *a, **k: None
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.shutdown()
        srv.server_close()


class HandRenderer:
    """A headless page that turns a pose into a PNG, recycled between batches."""

    def __init__(self, page_url: str, headless: bool = True):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=headless,
            args=["--use-gl=angle", "--enable-unsafe-swiftshader",
                  "--disable-dev-shm-usage"],
        )
        self.url = page_url
        self.page = None
        self.open()

    def open(self):
        if self.page is not None:
            return
        self.page = self._browser.new_page(viewport={"width": 600, "height": 820})
        self.page.goto(self.url)
        self.page.wait_for_function("window.__ready === true", timeout=60000)

    def recycle(self):
        """Close and reopen the page, so GPU buffers cannot accumulate."""
        if self.page is not None:
            self.page.close()
            self.page = None
        self.open()

    def frame_all(self, poses):
        """Fix the camera over every pose at once, so none can clip."""
        return self.page.evaluate("(ps) => window.frameAll(ps)", poses)

    def bones(self):
        return self.page.evaluate("window.__bones")

    def render(self, pose: dict, color: int) -> bytes:
        data = self.page.evaluate(
            "([p, c]) => window.renderPose(p, c)", [pose, color])
        return base64.b64decode(re.sub(r"^data:image/png;base64,", "", data))

    def close(self):
        try:
            if self.page is not None:
                self.page.close()
            self._browser.close()
        finally:
            self._pw.stop()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--page", type=Path, default=Path("assets/render/hand.html"))
    ap.add_argument("--out", type=Path, default=Path("assets/render/poses"))
    ap.add_argument("--sheet", action="store_true", help="render the 8 reference poses")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    with static_server(Path("assets")) as origin:
        url = f"{origin}/render/{args.page.name}"
        run_sheet(url, args)


def run_sheet(url: str, args) -> None:
    r = HandRenderer(url)
    try:
        r.frame_all([POSES[k] for k in ORDER])
        names = r.bones()
        have = [n for n in names if "finger" in n or "thumb" in n or n == "wrist"]
        print(f"rig exposes {len(have)} hand joints, e.g. {have[:4]}")
        if args.sheet:
            for key in ORDER:
                t0 = time.time()
                png = r.render(POSES[key], NEUTRAL)
                p = args.out / f"pose_{key}.png"
                p.write_bytes(png)
                print(f"  {key:<4} {len(png)/1024:6.0f} KB  {time.time()-t0:.2f}s")
    finally:
        r.close()


if __name__ == "__main__":
    main()
