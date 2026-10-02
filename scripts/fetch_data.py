"""Download and unpack NinaPro DB2 / DB3 into `data/raw/`.

The databases are served as one zip per subject, with no account and no login.
The two path conventions differ in more than the number, which is the kind of
thing that is easier to encode once than to retype:

    DB2   https://ninapro.hevs.ch/files/DB2_Preproc/DB2_s<N>.zip      N = 1..40
    DB3   https://ninapro.hevs.ch/files/db3_Preproc/s<N>_0.zip        N = 1..11
    DB6   https://ninapro.hevs.ch/files/DB6_Preproc/DB6_s<N>_{a,b}.zip  N = 1..10

DB6 ships **two** zips per subject rather than one, which is why `urls` returns a
list. Its sizes were measured by HEAD request rather than estimated: 21.80 GB for
the whole cohort, 2.18 GB per subject, the `_a` part about 1.3 GB and `_b` about
0.88 GB. DB2 is roughly 19 GB in total and DB3 roughly 2.8 GB, so:

  * Transfers resume. Re-running after an interruption picks up where it left
    off rather than starting the file again.
  * A subject whose `.mat` files are already unpacked is skipped entirely.
  * Zips are deleted once unpacked, so peak disk is one zip plus the extracted
    data rather than both in full. Pass --keep-zips to override.

Usage:

    uv run python scripts/fetch_data.py --dataset DB3
    uv run python scripts/fetch_data.py --dataset DB2 --subjects 1-4
    uv run python scripts/fetch_data.py --dataset DB2 --subjects 1,5,9 --keep-zips

Each zip is deleted as soon as it is unpacked, not at the end of the subject, so
peak disk for DB6 is one 1.3 GB zip plus the extracted data rather than both parts
at once.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

BASE = "https://ninapro.hevs.ch/files"

DATASETS = {
    "DB2": {
        "subjects": range(1, 41),
        "urls": lambda n: [f"{BASE}/DB2_Preproc/DB2_s{n}.zip"],
        "approx_mb": 470,
        # DB2/DB3 ship three exercise files per subject.
        "expect_files": 3,
        "patterns": lambda n: [f"S{n}_E*.mat", f"s{n}_E*.mat"],
    },
    "DB3": {
        "subjects": range(1, 12),
        "urls": lambda n: [f"{BASE}/db3_Preproc/s{n}_0.zip"],
        "approx_mb": 252,
        "expect_files": 3,
        "patterns": lambda n: [f"S{n}_E*.mat", f"s{n}_E*.mat"],
    },
    "DB6": {
        "subjects": range(1, 11),
        # Two parts per subject: 5 days x 2 trials, split across _a and _b.
        "urls": lambda n: [f"{BASE}/DB6_Preproc/DB6_s{n}_a.zip",
                           f"{BASE}/DB6_Preproc/DB6_s{n}_b.zip"],
        "approx_mb": 2180,          # measured by HEAD request, not estimated
        # 5 days x 2 trials. Requiring all ten means a half-finished subject is
        # re-fetched rather than quietly treated as complete, which matters more
        # here than for DB2: a missing day is a missing experimental condition.
        "expect_files": 10,
        "patterns": lambda n: [f"S{n}_D*.mat", f"s{n}_D*.mat"],
    },
}


def parse_subjects(spec: str | None, available: range) -> list[int]:
    """Parse "1-4", "1,3,5" or "1-4,7" into a sorted subject list."""
    if not spec:
        return list(available)
    picked: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            picked.update(range(int(lo), int(hi) + 1))
        else:
            picked.add(int(part))
    unknown = sorted(s for s in picked if s not in available)
    if unknown:
        raise SystemExit(
            f"subjects {unknown} are not in this dataset "
            f"(valid: {available.start}..{available.stop - 1})"
        )
    return sorted(picked)


def already_unpacked(dest: Path, subject: int, spec: dict) -> bool:
    """True when this subject's .mat files are all already on disk.

    Matches the loader's own filename conventions rather than a marker file, so a
    partially-manual download is recognised too. The expected count is per dataset:
    for DB6 an incomplete subject is a missing recording *day*, which is a missing
    experimental condition rather than a little less data, so it must not be
    mistaken for a finished one.
    """
    found = {p.name for pat in spec["patterns"](subject) for p in dest.rglob(pat)}
    return len(found) >= spec["expect_files"]


def download(url: str, path: Path, attempts: int = 8) -> None:
    """Fetch `url` to `path`, resuming a partial file if one is there.

    The server drops long transfers (curl 56, "failure receiving network data")
    and throttles sustained ones to tens of KB/s, so a single curl invocation is
    not enough to get a 250 MB file. Each attempt resumes from what is already on
    disk, and `--speed-limit` aborts a transfer that has genuinely stalled rather
    than letting it hang for hours at a few bytes a second.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl", "--location", "--fail", "--continue-at", "-",
        "--retry", "5", "--retry-delay", "5", "--retry-all-errors",
        "--speed-limit", "1024", "--speed-time", "120",
        "--progress-bar", "--output", str(path), url,
    ]

    for attempt in range(1, attempts + 1):
        before = path.stat().st_size if path.exists() else 0
        result = subprocess.run(cmd)
        if result.returncode == 0:
            return

        # curl 33: the server refused a range request, so resuming is impossible
        # and the partial file has to go.
        if result.returncode == 33 and path.exists():
            print("    server refused to resume; restarting this file", flush=True)
            path.unlink()
            continue

        after = path.stat().st_size if path.exists() else 0
        gained = after - before
        if attempt < attempts:
            print(f"    curl exit {result.returncode} after {gained / 1e6:.1f} MB "
                  f"(attempt {attempt}/{attempts}); resuming from {after / 1e6:.0f} MB",
                  flush=True)
            time.sleep(min(60, 5 * attempt))

    raise RuntimeError(f"download failed after {attempts} attempts: {url}")


def unpack(zip_path: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if not m.startswith("__MACOSX")]
            zf.extractall(dest, members=members)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"{zip_path.name} is not a readable zip ({exc}). The download was "
            "probably truncated -- delete it and re-run to fetch it again."
        ) from exc
    return len(members)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--subjects", help='e.g. "1-4", "1,3,5", "1-4,7". Default: all.')
    ap.add_argument("--out", default="data/raw", type=Path)
    ap.add_argument("--keep-zips", action="store_true",
                    help="keep the downloaded zips instead of deleting them once unpacked")
    args = ap.parse_args()

    if shutil.which("curl") is None:
        raise SystemExit("curl not found on PATH; install it or download the zips by hand")

    spec = DATASETS[args.dataset]
    wanted = parse_subjects(args.subjects, spec["subjects"])
    dest = args.out / args.dataset
    staging = args.out / "_downloads"

    est_gb = len(wanted) * spec["approx_mb"] / 1024
    print(f"{args.dataset}: {len(wanted)} subject(s), roughly {est_gb:.1f} GB -> {dest}",
          flush=True)

    fetched = skipped = 0
    failed: list[tuple[int, str]] = []
    for subject in wanted:
        if already_unpacked(dest, subject, spec):
            print(f"  S{subject:<3} already unpacked, skipping", flush=True)
            skipped += 1
            continue

        urls = spec["urls"](subject)
        # One subject failing must not end the run -- the remaining subjects are
        # independent, and a partial cohort is still worth having on disk.
        try:
            n = 0
            for url in urls:
                zip_path = staging / Path(url).name
                print(f"  S{subject:<3} {url}", flush=True)
                download(url, zip_path)
                n += unpack(zip_path, dest)
                # Deleted per part, not per subject, so two 1.3 GB zips never sit
                # on disk together.
                if not args.keep_zips:
                    zip_path.unlink()
        except (RuntimeError, OSError) as exc:
            print(f"    FAILED: {exc}", flush=True)
            failed.append((subject, str(exc)))
            continue
        print(f"    unpacked {n} file(s)", flush=True)
        fetched += 1

    if staging.exists() and not any(staging.iterdir()):
        staging.rmdir()

    print(f"\n{fetched} fetched, {skipped} already present, {len(failed)} failed.")
    if failed:
        print("failed subjects (re-run the same command to retry just these):")
        for subject, why in failed:
            print(f"  S{subject}: {why}")
    print(f"Next: uv run python scripts/inspect_data.py {dest} --dataset {args.dataset}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted -- re-run the same command to resume", file=sys.stderr)
        sys.exit(130)
