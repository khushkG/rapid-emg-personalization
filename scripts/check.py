"""One command that answers "is this project working?".

    uv run python scripts/check.py            # ~4 minutes, no data needed
    uv run python scripts/check.py --quick    # ~20 seconds, tests only
    uv run python scripts/check.py --full     # also runs a real experiment

Four levels, cheapest first, each answering a different question:

  1. tests        Is the logic right? Runs without any data at all.
  2. synthetic    Do the stages connect? Trains and adapts on simulated EMG.
  3. data         Do the real recordings match what the loader assumes?
  4. experiment   Does it produce results on real amputee data? (--full)

The point of the ladder is that a failure tells you *where* the problem is. A
green level 1 with a red level 3 means the code is fine and the data is not,
which is a different morning's work from the reverse.

Levels are skipped, not failed, when their inputs are absent -- a missing DB2
download is a thing you have not done yet, not a broken project -- and the
summary says plainly which is which.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

OK, FAIL, SKIP = "PASS", "FAIL", "SKIP"


class Result:
    def __init__(self, name: str, status: str, detail: str = "", seconds: float = 0.0):
        self.name, self.status, self.detail, self.seconds = name, status, detail, seconds


def run(cmd: list[str], timeout: int = 1800) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"


def check_tests() -> Result:
    t0 = time.time()
    code, out = run(["uv", "run", "--group", "dev", "pytest", "-q"])
    tail = [ln for ln in out.strip().splitlines() if ln.strip()]
    summary = tail[-1] if tail else "no output"
    return Result("1. tests", OK if code == 0 else FAIL, summary, time.time() - t0)


def check_synthetic() -> Result:
    t0 = time.time()
    code, out = run(["uv", "run", "python", "scripts/smoke.py"])
    if code != 0:
        last = out.strip().splitlines()[-3:]
        return Result("2. synthetic end-to-end", FAIL, " | ".join(last), time.time() - t0)
    # The smoke test is plumbing, but one thing is still meaningful: adapting to
    # a subject should beat not adapting. If it does not, something is wired
    # wrong even though every stage "ran".
    rows = [ln for ln in out.splitlines() if ln.strip().startswith(("1 ", "3 "))]
    none_rows = [ln for ln in rows if " none " in ln]
    best = [ln for ln in rows if " rapid " in ln or " linear_probe " in ln]
    detail = "ran; "
    try:
        worst_none = max(float(ln.split()[2]) for ln in none_rows)
        best_adapted = max(float(ln.split()[2]) for ln in best)
        if best_adapted <= worst_none:
            return Result("2. synthetic end-to-end", FAIL,
                          f"adapting ({best_adapted:.3f}) did not beat not adapting "
                          f"({worst_none:.3f}) even on synthetic data",
                          time.time() - t0)
        detail += f"adaptation helps ({worst_none:.3f} -> {best_adapted:.3f} bal acc)"
    except (ValueError, IndexError):
        detail += "could not parse the summary table"
    return Result("2. synthetic end-to-end", OK, detail, time.time() - t0)


def check_data(root: Path) -> list[Result]:
    out = []
    for dataset in ("DB2", "DB3"):
        path = root / dataset
        subjects = len(list(path.glob("*"))) if path.exists() else 0
        if not subjects:
            out.append(Result(f"3. data: {dataset}", SKIP,
                              f"not downloaded -- uv run python scripts/fetch_data.py "
                              f"--dataset {dataset}"))
            continue
        t0 = time.time()
        code, text = run(["uv", "run", "python", "scripts/inspect_data.py",
                          str(path), "--dataset", dataset])
        if code != 0:
            out.append(Result(f"3. data: {dataset}", FAIL,
                              text.strip().splitlines()[-1] if text.strip() else "failed",
                              time.time() - t0))
            continue
        clean = "no structural problems found" in text
        note = f"{subjects} subject(s); "
        note += "no structural problems" if clean else "PROBLEMS REPORTED - read the output"
        if "NOT in every subject" in text:
            note += "; some movements missing from some subjects"
        out.append(Result(f"3. data: {dataset}", OK if clean else FAIL, note,
                          time.time() - t0))
    return out


def check_experiment(root: Path) -> Result:
    if not (root / "DB2").exists() or not (root / "DB3").exists():
        return Result("4. real experiment", SKIP, "needs both DB2 and DB3 downloaded")
    t0 = time.time()
    code, out = run(["uv", "run", "python", "scripts/run_experiment.py",
                     "--steps", "400", "--shots", "1", "--tag", "selfcheck", "--no-notch"],
                    timeout=3600)
    if code != 0:
        return Result("4. real experiment", FAIL,
                      out.strip().splitlines()[-1] if out.strip() else "failed",
                      time.time() - t0)
    return Result("4. real experiment", OK,
                  "ran on real data; see results/selfcheck_rows.csv "
                  "(400 steps -- a smoke test, not a result)", time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="tests only (~20s)")
    ap.add_argument("--full", action="store_true", help="also run a real experiment")
    ap.add_argument("--data", type=Path, default=Path("data/raw"))
    args = ap.parse_args()

    if shutil.which("uv") is None:
        raise SystemExit("uv not found on PATH")

    print("Checking the project. Each level answers a different question.\n", flush=True)
    results = [check_tests()]
    print(f"  {results[-1].status}  {results[-1].name}  ({results[-1].seconds:.0f}s)",
          flush=True)

    if not args.quick:
        for r in [check_synthetic(), *check_data(args.data)]:
            results.append(r)
            print(f"  {r.status}  {r.name}  ({r.seconds:.0f}s)", flush=True)
        if args.full:
            r = check_experiment(args.data)
            results.append(r)
            print(f"  {r.status}  {r.name}  ({r.seconds:.0f}s)", flush=True)

    print("\n" + "-" * 72)
    for r in results:
        print(f"{r.status:<5} {r.name:<26} {r.detail}")
    print("-" * 72)

    failed = [r for r in results if r.status == FAIL]
    skipped = [r for r in results if r.status == SKIP]
    if failed:
        print(f"\n{len(failed)} level(s) FAILED. The lowest failing level is where to look.")
        sys.exit(1)
    print(f"\nAll {len(results) - len(skipped)} level(s) checked passed."
          + (f" {len(skipped)} skipped for missing data." if skipped else ""))
    if skipped:
        print("A skipped level is work not yet done, not a broken project.")


if __name__ == "__main__":
    main()
