"""Build the single self-contained project page.

Everything is inlined: the two figures as base64 PNGs, the hand geometry as SVG path
data, and the decoded label sequences as JSON. The page opens from disk with no
server, no network and no libraries.

It contains no EMG. NinaPro publishes a citation requirement and no redistribution
licence, so what travels in this file is model predictions, cued labels and aggregate
metrics -- none of which allows a signal to be reconstructed.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from remg.data.movements import DB6_GRASP_SET
from remg.viz.hand import POSES, pose_geometry

# Must match scripts/make_gif.py -- arbitrary, maximally distinguishable, and not a
# claim about grasp identity (no source states which id is which).
DB6_POSE = {
    "rest": "rest",
    "db6_movement_1": "close_hand",
    "db6_movement_3": "open_hand",
    "db6_movement_4": "point_index",
    "db6_movement_6": "wrist_flexion",
    "db6_movement_9": "wrist_extension",
    "db6_movement_10": "wrist_supination",
    "db6_movement_11": "wrist_pronation",
}
METHODS = [
    ("finetune", "standard fine-tuning"),
    ("rapid", "rapid personalization"),
    ("linear_probe", "linear probe"),
    ("td_rf", "classic features (random forest)"),
    ("finetune_classic_gate", "fine-tuning + classic rest gate"),
    ("none", "no personalization"),
]
SUBJECT_NOTE = {
    1: "S1 — barely drifts between days",
    2: "S2 — drifts the most of the ten",
    6: "S6 — closest to the cohort average",
}

VB = (-1.55, -1.25, 3.10, 3.10)          # x, y, w, h in hand coordinates


def svg_paths(pose_name: str) -> dict:
    """Pose geometry as SVG path strings, y flipped for screen coordinates."""
    geo = pose_geometry(POSES[pose_name])

    def path(pts: np.ndarray, close: bool) -> str:
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.3f},{-y:.3f}"
                     for i, (x, y) in enumerate(pts))
        return d + (" Z" if close else "")

    return {"palm": path(geo["palm"], True),
            "fingers": [path(geo[f], False)
                        for f in ("thumb", "index", "middle", "ring", "little")]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", type=Path, default=Path("results/visuals/preds_db6.csv"))
    ap.add_argument("--summary", type=Path,
                    default=Path("results/visuals/preds_db6_summary.csv"))
    ap.add_argument("--meta", type=Path,
                    default=Path("results/visuals/preds_db6_meta.json"))
    ap.add_argument("--chart", type=Path, default=Path("results/visuals/day1_to_day5.png"))
    ap.add_argument("--timeline", type=Path,
                    default=Path("results/visuals/rest_timeline.png"))
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 6])
    ap.add_argument("--frames", type=int, default=220)
    ap.add_argument("--out", type=Path, default=Path("results/project_page.html"))
    args = ap.parse_args()

    classes = json.loads(args.meta.read_text())["classes"]
    d = pd.read_csv(args.preds)
    summ = pd.read_csv(args.summary)

    poses = {c: svg_paths(DB6_POSE[c]) for c in classes}
    short = {c: ("rest" if c == "rest" else f"movement {c.split('_')[-1]}")
             for c in classes}

    seqs: dict = {}
    accs: dict = {}
    for subj in args.subjects:
        for day in ("day1", "day5"):
            q = d[(d.subject == subj) & (d.day == day)].sort_values("start")
            # A stretch with several movements and some rest, so the viewer sees the
            # hand change rather than one long hold.
            yt = q.y_true.to_numpy()
            best, score = 0, -1.0
            for st in range(0, max(1, len(yt) - args.frames), 7):
                seg = yt[st:st + args.frames]
                if len(seg) < args.frames:
                    break
                sc = len(np.unique(seg)) - 6.0 * abs(float((seg == 0).mean()) - 0.35)
                if sc > score:
                    best, score = st, sc
            seg = q.iloc[best:best + args.frames]
            key = f"{subj}_{day}"
            seqs[key] = {"true": [int(v) for v in seg.y_true]}
            for m, _ in METHODS:
                col = f"pred_{m}"
                if col in seg.columns:
                    seqs[key][m] = [int(v) for v in seg[col]]
            a = summ[(summ.subject == subj) & (summ.day == day)]
            accs[key] = {r.method: round(float(r.balanced_accuracy), 3)
                         for r in a.itertuples()}

    def b64(p: Path) -> str:
        return base64.b64encode(p.read_bytes()).decode("ascii")

    data = {"classes": classes, "short": short, "poses": poses, "seqs": seqs,
            "accs": accs, "methods": METHODS,
            "subjects": [[s, SUBJECT_NOTE.get(s, f"S{s}")] for s in args.subjects],
            "vb": list(VB), "grasps": list(DB6_GRASP_SET)}

    html = PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    html = html.replace("__CHART__", b64(args.chart))
    html = html.replace("__TIMELINE__", b64(args.timeline))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1024 / 1024:.2f} MB)")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Can a hand-control model learn a new person from three examples?</title>
<style>
:root{--ink:#0b0b0b;--ink2:#52514e;--surface:#fcfcfb;--line:#e6e6e3;
--ok:#1f8f5f;--bad:#d4351c;--accent:#2a78d6}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);
font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:48px 20px 80px}
h1{font-size:clamp(26px,4.4vw,40px);line-height:1.18;margin:0 0 14px;letter-spacing:-.01em}
h2{font-size:clamp(19px,2.6vw,24px);margin:52px 0 10px;letter-spacing:-.005em}
p{margin:0 0 15px;max-width:68ch}
.lede{font-size:clamp(17px,2.1vw,19px);color:var(--ink2)}
.muted{color:var(--ink2)}
.small{font-size:13.5px;color:var(--ink2)}
figure{margin:22px 0 10px}
img{max-width:100%;height:auto;display:block;border:1px solid var(--line);border-radius:8px}
figcaption{font-size:13.5px;color:var(--ink2);margin-top:9px;max-width:72ch}
.controls{display:flex;flex-wrap:wrap;gap:16px;margin:18px 0 10px;align-items:flex-end}
.ctl{display:flex;flex-direction:column;gap:5px;min-width:0}
label{font-size:12.5px;color:var(--ink2);text-transform:uppercase;letter-spacing:.055em}
select,button{font:inherit;font-size:14.5px;padding:8px 11px;border:1px solid var(--line);
border-radius:7px;background:#fff;color:var(--ink);max-width:100%}
button{cursor:pointer}
button:hover{border-color:#c9c9c4}
.stage{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:8px}
.panel{border:1px solid var(--line);border-radius:10px;padding:16px;background:#fff;min-width:0}
.panel h3{margin:0 0 2px;font-size:15px;font-weight:600}
.panel .sub{font-size:13px;color:var(--ink2);margin-bottom:8px}
svg{width:100%;height:auto;display:block}
.verdict{font-size:14px;font-weight:600;margin-top:6px;min-height:1.5em}
.scrub{width:100%;margin-top:14px}
table{border-collapse:collapse;width:100%;font-size:14.5px;margin:10px 0 6px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}
th{font-weight:600;color:var(--ink2);font-size:12.5px;text-transform:uppercase;
letter-spacing:.05em}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.note{border-left:3px solid var(--line);padding:2px 0 2px 15px;margin:20px 0;
color:var(--ink2);font-size:14.5px}
@media (max-width:720px){.stage{grid-template-columns:1fr}}
</style></head><body><div class="wrap">

<h1>Can a hand-control model learn a new person from three examples?</h1>
<p class="lede">A prosthetic hand reads electrical signals from the muscles left in the
forearm and has to guess what movement the wearer intends. Every person's muscles and
every fitting of the electrodes are different, so a model trained on other people
starts out poor and has to be calibrated. This project asked how little calibration
you can get away with — and what breaks when the person takes the sleeve off and puts
it back on days later.</p>

<p>Everything below comes from public recordings (NinaPro) and from models trained
here. The honest summary is that personalization works, the flashy method does not
beat the plain ones, and the thing that actually matters for a wearable — not moving
when the user is holding still — is where every deep model does worst.</p>

<h2>What happens five days later</h2>
<p>A model is calibrated once, from three repetitions on day 1. It is then tested on
day 1 and again on day 5, after the electrodes have been taken off and put back on.
The gap between the two is the part nobody sees in a single-session demo.</p>
<figure><img src="data:image/png;base64,__CHART__"
  alt="Line chart of balanced accuracy on day 1 and day 5 for five methods. All
  methods fall; standard fine-tuning falls most, from 0.389 to 0.281. Rapid
  personalization falls least among the adapted methods, 0.412 to 0.339.">
<figcaption>Fitting every weight in the network to one session's electrode placement
is what degrades most. Methods that change fewer numbers hold up better — and a plain
linear probe holds up about as well as the elaborate adapter method it was meant to
lose to.</figcaption></figure>

<h2>Moving when it should be still</h2>
<p>Accuracy hides the error that matters most. A prosthesis that confuses two grasps
is annoying; one that moves while you are holding a glass is unusable. Below, a
minute of real recording: pale is the user at rest, blue-grey is a genuine movement,
and every red mark is the model commanding a movement that was never intended.</p>
<figure><img src="data:image/png;base64,__TIMELINE__"
  alt="Two timelines showing a minute of recording. The upper row, standard
  fine-tuning, has frequent red marks during rest. The lower row adds a classic rest
  gate and has far fewer.">
<figcaption>Adding a simple classifier whose only job is to decide <em>whether</em> to
move at all removes most of the unwanted activations. It cost almost no
accuracy.</figcaption></figure>

<h2>Watch it decode</h2>
<p>The left hand in each panel is the movement the person was asked to make; the right
is what the model actually commanded. <span style="color:var(--ok);font-weight:600">
Green</span> means it matched,
<span style="color:var(--bad);font-weight:600">red</span> means it did not. Pick a
person, a method, and a day.</p>

<div class="controls">
  <div class="ctl"><label for="subj">Person</label><select id="subj"></select></div>
  <div class="ctl"><label for="meth">Method</label><select id="meth"></select></div>
  <div class="ctl"><label for="day">Day</label><select id="day">
    <option value="day1">Day 1 — same session as calibration</option>
    <option value="day5">Day 5 — electrodes re-donned</option>
  </select></div>
  <div class="ctl"><label for="play">&nbsp;</label>
    <button id="play">Pause</button></div>
</div>

<div class="stage">
  <div class="panel"><h3>Cued</h3><div class="sub" id="cue-name">—</div>
    <svg id="cue" viewBox="0 0 1 1" aria-label="the movement the person was asked to make"></svg></div>
  <div class="panel"><h3>Decoded</h3><div class="sub" id="dec-name">—</div>
    <svg id="dec" viewBox="0 0 1 1" aria-label="the movement the model commanded"></svg>
    <div class="verdict" id="verdict"></div></div>
</div>
<input class="scrub" id="scrub" type="range" min="0" max="1" value="0" step="1"
  aria-label="scrub through the recording">
<p class="small" id="acc-line"></p>

<h2>The numbers</h2>
<table id="tbl"><thead><tr><th>Method</th><th class="num">Day 1</th>
<th class="num">Day 5</th><th class="num">Drop</th></tr></thead><tbody></tbody></table>
<p class="small">Balanced accuracy for the selected person, from the same single
calibration. Chance is 0.125.</p>

<div class="note">
<strong>What this page does not show.</strong> No muscle signal appears anywhere here,
and none is embedded in this file. NinaPro asks that its papers be cited and publishes
no licence permitting redistribution of the recordings, so everything above is model
output, cued labels, and summary numbers.<br><br>
<strong>About the hand shapes.</strong> DB6's seven grasps are Large Diameter, Adducted
Thumb, Index Finger Extension, Medium Wrap, Writing Tripod, Power Sphere and Precision
Sphere (Palermo et al., IEEE ICORR 2017). No source consulted states which numeric
label corresponds to which grasp, so the shapes drawn here are arbitrary but
consistently distinguishable placeholders — chosen so you can see the decoded hand
change, not to depict a particular grasp.<br><br>
<strong>Data.</strong> Atzori et al., <em>Scientific Data</em> 2014 (DB2, DB3);
Palermo et al., IEEE ICORR 2017 (DB6).
</div>

<script>
const D = __DATA__;
const $ = s => document.querySelector(s);
const subjSel = $("#subj"), methSel = $("#meth"), daySel = $("#day");
D.subjects.forEach(([s, note]) => subjSel.add(new Option(note, s)));
D.methods.forEach(([k, label]) => { if (anySeq(k)) methSel.add(new Option(label, k)); });
function anySeq(k){ return Object.values(D.seqs).some(v => v[k]); }
subjSel.value = "2"; methSel.value = "finetune"; daySel.value = "day5";

const vb = D.vb;
for (const id of ["cue", "dec"]) $("#"+id).setAttribute("viewBox", vb.join(" "));

function drawHand(svg, cls, color){
  const p = D.poses[cls];
  const parts = [`<path d="${p.palm}" fill="${color?color:"#ccd4e0"}"
     stroke="#2b3a4a" stroke-width="0.035" opacity="${color?0.92:1}"/>`];
  for (const f of p.fingers)
    parts.push(`<path d="${f}" fill="none" stroke="${color?color:"#2b3a4a"}"
       stroke-width="0.17" stroke-linecap="round" stroke-linejoin="round"/>`);
  svg.innerHTML = parts.join("");
}

let i = 0, timer = null, playing = true;
function key(){ return subjSel.value + "_" + daySel.value; }
function frames(){ const s = D.seqs[key()]; return s ? s.true.length : 0; }

function render(){
  const s = D.seqs[key()]; if (!s) return;
  const m = methSel.value, pred = s[m] || s.true;
  const n = s.true.length; i = ((i % n) + n) % n;
  const t = D.classes[s.true[i]], p = D.classes[pred[i]];
  const ok = t === p;
  drawHand($("#cue"), t, null);
  drawHand($("#dec"), p, ok ? "#1f8f5f" : "#d4351c");
  $("#cue-name").textContent = D.short[t];
  $("#dec-name").textContent = D.short[p];
  $("#dec-name").style.color = ok ? "#1f8f5f" : "#d4351c";
  $("#verdict").textContent = ok ? "correct" : "wrong";
  $("#verdict").style.color = ok ? "#1f8f5f" : "#d4351c";
  $("#scrub").max = n - 1; $("#scrub").value = i;
  const a = (D.accs[key()] || {})[m];
  $("#acc-line").textContent = a === undefined ? "" :
    `Balanced accuracy over the whole of this day's recording: ${a.toFixed(3)}` +
    ` · frame ${i + 1} of ${n} · each frame is 100 ms of recording`;
}

function table(){
  const body = $("#tbl tbody"); body.innerHTML = "";
  const a1 = D.accs[subjSel.value + "_day1"] || {},
        a5 = D.accs[subjSel.value + "_day5"] || {};
  for (const [k, label] of D.methods){
    if (a1[k] === undefined && a5[k] === undefined) continue;
    const drop = (a1[k] !== undefined && a5[k] !== undefined)
      ? (a1[k] - a5[k]) : null;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${label}</td><td class="num">${fmt(a1[k])}</td>` +
      `<td class="num">${fmt(a5[k])}</td><td class="num">` +
      (drop === null ? "—" : (drop >= 0 ? "+" : "") + drop.toFixed(3)) + `</td>`;
    if (k === methSel.value) tr.style.background = "#f4f7fb";
    body.appendChild(tr);
  }
}
const fmt = v => v === undefined ? "—" : v.toFixed(3);

function tick(){ if (playing) { i++; render(); } }
function restart(){ i = 0; render(); table(); }
subjSel.onchange = restart; daySel.onchange = restart;
methSel.onchange = () => { render(); table(); };
$("#scrub").oninput = e => { playing = false; $("#play").textContent = "Play";
  i = +e.target.value; render(); };
$("#play").onclick = () => { playing = !playing;
  $("#play").textContent = playing ? "Pause" : "Play"; };
restart();
timer = setInterval(tick, 100);
</script>
</div></body></html>
"""

if __name__ == "__main__":
    main()
