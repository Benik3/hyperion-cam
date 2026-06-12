#!/usr/bin/env python3
"""Generate the LED-calibration video for hyperion-cam --ledcalib.

Design: doc/led-lut-calibration.md. The video has three sections:
  1. white hold        - phase 0 (eye anchor): full-screen white
  2. TV patch sweep    - phase 1: each patch preceded by a 3-digit base-6
                         full-screen color index code
  3. black hold        - phase 2 (LED sweep): TV must stay black

The digit colors and nominal timings MUST match the decoder in
src/ledCalib.cpp (kPatchWindowDelayMs and LedCalib::digitValue).

Requires: ffmpeg on PATH (stdlib-only Python).

Usage:
  tools/gen-calib-video.py --out calib.mp4
  # then play calib.mp4 on the TV (USB stick / media app) during --ledcalib
"""

import argparse
import json
import shutil
import subprocess
import sys

# Base-6 digit colors, value = list index. Must match LedCalib::digitValue():
# WHITE=0 RED=1 GREEN=2 BLUE=3 YELLOW=4 CYAN=5
DIGIT_COLORS = [
    (255, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
]

# Nominal timings (seconds); the decoder derives its patch measurement window
# from these. Change them only together with src/ledCalib.cpp.
T_SYNC_WHITE = 0.4
T_SYNC_BLACK = 0.3
T_DIGIT = 0.4
T_DIGIT_GAP = 0.2
T_PRE_PATCH = 0.3
T_PATCH = 1.5
T_POST_PATCH = 0.3

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def patch_list(grid):
    levels = [round(i * 255 / (grid - 1)) for i in range(grid)]
    patches = []
    for ri in range(grid):
        for gi in range(grid):
            for bi in range(grid):
                idx = ri * grid * grid + gi * grid + bi
                patches.append((idx, (levels[ri], levels[gi], levels[bi])))
    return patches


def index_digits(idx):
    return (idx // 36, (idx // 6) % 6, idx % 6)


def build_sequence(args):
    """Return ([(rgb, seconds)], manifest_dict)."""
    seq = [(WHITE, args.white_hold)]
    patches = patch_list(args.grid)

    sweep_start = args.white_hold
    for idx, rgb in patches:
        d1, d2, d3 = index_digits(idx)
        seq += [
            (WHITE, T_SYNC_WHITE),
            (BLACK, T_SYNC_BLACK),
            (DIGIT_COLORS[d1], T_DIGIT),
            (BLACK, T_DIGIT_GAP),
            (DIGIT_COLORS[d2], T_DIGIT),
            (BLACK, T_DIGIT_GAP),
            (DIGIT_COLORS[d3], T_DIGIT),
            (BLACK, T_PRE_PATCH),
            (rgb, T_PATCH),
            (BLACK, T_POST_PATCH),
        ]

    sweep_end = sum(t for _, t in seq)
    seq.append((BLACK, args.black_hold))

    manifest = {
        "fps": args.fps,
        "size": [args.width, args.height],
        "grid": args.grid,
        "patch_count": len(patches),
        "timings": {
            "sync_white": T_SYNC_WHITE,
            "sync_black": T_SYNC_BLACK,
            "digit": T_DIGIT,
            "digit_gap": T_DIGIT_GAP,
            "pre_patch": T_PRE_PATCH,
            "patch": T_PATCH,
            "post_patch": T_POST_PATCH,
        },
        "sections": {
            "white_hold": [0.0, args.white_hold],
            "tv_sweep": [sweep_start, sweep_end],
            "black_hold": [sweep_end, sweep_end + args.black_hold],
        },
        "patches": [{"index": idx, "rgb": list(rgb)} for idx, rgb in patches],
    }
    return seq, manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="calib.mp4", help="output video file (default calib.mp4)")
    ap.add_argument("--manifest", default=None, help="manifest JSON path (default <out>.manifest.json)")
    ap.add_argument("--grid", type=int, default=6, help="patch grid per axis, 2-6 (6^3=216 = index-code limit)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--white-hold", type=float, default=300.0, help="seconds of white for the eye-anchor phase")
    ap.add_argument("--black-hold", type=float, default=1200.0, help="seconds of black for the LED sweep phase")
    ap.add_argument("--crf", type=int, default=18, help="x264 quality (lower = better)")
    args = ap.parse_args()

    if not 2 <= args.grid <= 6:
        sys.exit("ERROR: --grid must be 2-6 (3 base-6 digits address at most 216 patches)")
    if shutil.which("ffmpeg") is None:
        sys.exit("ERROR: ffmpeg not found on PATH")

    seq, manifest = build_sequence(args)
    total = sum(t for _, t in seq)
    n_patches = manifest["patch_count"]
    print(f"{n_patches} patches, total video {total / 60:.1f} min "
          f"(TV sweep {(manifest['sections']['tv_sweep'][1] - manifest['sections']['tv_sweep'][0]) / 60:.1f} min)")

    manifest_path = args.manifest or args.out + ".manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {manifest_path}")

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        args.out,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # Solid frames only: render each color once, repeat its bytes. Durations are
    # accumulated in frame counts so rounding never drifts the nominal timings.
    emitted = 0
    t_acc = 0.0
    for rgb, dur in seq:
        t_acc += dur
        n_target = round(t_acc * args.fps)
        frame = bytes(rgb) * (args.width * args.height)
        for _ in range(n_target - emitted):
            proc.stdin.write(frame)
        emitted = n_target

    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("ERROR: ffmpeg failed")
    print(f"wrote {args.out} ({emitted} frames)")


if __name__ == "__main__":
    main()
