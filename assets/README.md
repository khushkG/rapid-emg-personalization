# Third-party render assets

The hand animation is rendered with a real rigged 3D model rather than drawn, because
a 2D hand could not be made to read: a closed hand loses its fingers behind the palm,
and poses that differ anatomically share a silhouette from any one viewpoint.

## Hand model — committed here

`webxr_generic_hand_left.glb` (94 KB), the `generic-hand` profile from
[WebXR Input Profiles](https://github.com/immersive-web/webxr-input-profiles), by the
W3C Immersive Web Working Group.

**Licence: [W3C Software and Document License](https://www.w3.org/Consortium/Legal/copyright-software)**,
notice preserved in `LICENSE.webxr-input-profiles.md`.

Note for anyone auditing this: the npm package `@webxr-input-profiles/assets` declares
**no licence field** in its registry metadata for any version. The licence above comes
from the project's repository, not from the published artefact. That gap was raised
before the model was used and judged acceptable.

The rig has 25 standard XR hand joints. One thing about it is load-bearing and not
obvious: **the skeleton is flat** — every joint's parent is `Armature`, not the
preceding joint, because these models carry absolute tracked joint poses rather than a
kinematic chain. Rotating a knuckle moves nothing downstream. `assets/render/hand.html`
rebuilds the hierarchy with `Object3D.attach()` before posing anything.

## three.js — not committed

r160, MIT, with its notice in `three/LICENSE.three.txt`. The library itself is
gitignored; restore it with:

```
mkdir -p assets/three/build assets/three/examples/jsm/loaders assets/three/examples/jsm/utils
B=https://cdn.jsdelivr.net/npm/three@0.160.0
curl -sL -o assets/three/build/three.module.js $B/build/three.module.js
curl -sL -o assets/three/examples/jsm/loaders/GLTFLoader.js $B/examples/jsm/loaders/GLTFLoader.js
curl -sL -o assets/three/examples/jsm/utils/BufferGeometryUtils.js $B/examples/jsm/utils/BufferGeometryUtils.js
```

The directory layout matters: `GLTFLoader` imports `../utils/BufferGeometryUtils.js`
relative to itself, so a flat folder will 404.

## Rendering

```
uv run playwright install chromium                      # once
PYTHONPATH=scripts uv run python scripts/render_hands.py --sheet        # the 8 poses
PYTHONPATH=scripts uv run python scripts/render_pose_sprites.py         # sprites for the page
PYTHONPATH=scripts uv run python scripts/make_hand_gif3d.py             # the animation
```

Chromium refuses ES modules over `file://`, so the scripts serve `assets/` on a
loopback port for the duration of a render. Nothing leaves the machine.
