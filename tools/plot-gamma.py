#!/usr/bin/env python3
"""Plot LED-calib gamma ramps (doc/led-lut-calibration.md) as a standalone SVG.

Stdlib only — no numpy/matplotlib, runs anywhere including the Pi. Reads the
gamma.csv (ISP RGB) and/or raw_gamma.csv (RAW Bayer) written by
`hyperion-cam --ledcalib gamma`, and draws the measured response of each
channel vs the commanded code value. ISP and RAW are normalised to 0..1 and
overlaid so crush (flat low end) / clip (flat high end) and ISP-vs-RAW
divergence are obvious.

Usage:
  tools/plot-gamma.py ledcalib-out/            # auto-finds gamma.csv [+ raw_gamma.csv]
  tools/plot-gamma.py gamma.csv raw_gamma.csv -o out.svg
  open the .svg in any browser
"""

import argparse
import csv
import os
import sys

# channel -> the measured component to treat as the response, and a colour.
CH = {
    "R": (0, "#d22"),
    "G": (1, "#2a2"),
    "B": (2, "#36d"),
    "gray": (None, "#888"),  # gray: use luminance-ish mean of the 3
}
W, H = 900, 640
PAD = 60


def load(path):
    """Return {channel: [(level, measR, measG, measB), ...]} sorted by level."""
    out = {}
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    header = rows[0]
    idx = {name: i for i, name in enumerate(header)}
    for r in rows[1:]:
        ch = r[idx["channel"]]
        lvl = float(r[idx["level"]])
        m = (float(r[idx["meas_r"]]), float(r[idx["meas_g"]]), float(r[idx["meas_b"]]))
        out.setdefault(ch, []).append((lvl, m))
    for ch in out:
        out[ch].sort()
    return out


def series_value(ch, meas):
    comp, _ = CH[ch]
    if comp is None:
        return sum(meas) / 3.0
    return meas[comp]


def norm_max(datasets):
    """Global max measured value across all sets, per the dominant component."""
    mx = 1e-9
    for data in datasets:
        for ch, pts in data.items():
            for _, m in pts:
                mx = max(mx, series_value(ch, m))
    return mx


def line(x1, y1, x2, y2, col, w=1, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" stroke-width="{w}"{d}/>'


def polyline(pts, col, dash=""):
    p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{p}" fill="none" stroke="{col}" stroke-width="2"{d}/>'


def build_svg(isp, raw):
    datasets = [d for d in (isp, raw) if d]
    gmax = norm_max(datasets)

    def X(level):
        return PAD + (level / 255.0) * (W - 2 * PAD)

    def Y(val):
        return H - PAD - (val / gmax) * (H - 2 * PAD)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="13">']
    s.append(f'<rect width="{W}" height="{H}" fill="#fff"/>')
    # axes + grid
    for g in range(11):
        gx = PAD + g / 10.0 * (W - 2 * PAD)
        s.append(line(gx, PAD, gx, H - PAD, "#eee"))
        s.append(f'<text x="{gx:.0f}" y="{H-PAD+18:.0f}" text-anchor="middle" fill="#666">{g*255//10}</text>')
        gy = H - PAD - g / 10.0 * (H - 2 * PAD)
        s.append(line(PAD, gy, W - PAD, gy, "#eee"))
        s.append(f'<text x="{PAD-8:.0f}" y="{gy+4:.0f}" text-anchor="end" fill="#666">{g*10}%</text>')
    s.append(line(PAD, PAD, PAD, H - PAD, "#333"))
    s.append(line(PAD, H - PAD, W - PAD, H - PAD, "#333"))
    # diagonal reference (perfect linear) faint
    s.append(line(X(0), Y(0), X(255), Y(gmax), "#ccc", 1, "4 4"))
    s.append(f'<text x="{W/2:.0f}" y="{H-12:.0f}" text-anchor="middle" fill="#333">commanded code value</text>')
    s.append(f'<text x="16" y="{H/2:.0f}" text-anchor="middle" fill="#333" transform="rotate(-90 16 {H/2:.0f})">measured (% of max)</text>')

    def draw(data, dash, label_suffix):
        for ch, (_, col) in CH.items():
            if ch not in data:
                continue
            pts = [(X(lvl), Y(series_value(ch, m))) for lvl, m in data[ch]]
            s.append(polyline(pts, col, dash))

    if isp:
        draw(isp, "", " ISP")
    if raw:
        draw(raw, "6 3", " RAW")

    # legend
    ly = PAD + 6
    for ch, (_, col) in CH.items():
        s.append(f'<text x="{W-PAD-90:.0f}" y="{ly:.0f}" fill="{col}">{ch}</text>')
        ly += 18
    if isp and raw:
        s.append(f'<text x="{W-PAD-90:.0f}" y="{ly:.0f}" fill="#333">solid=ISP</text>')
        s.append(f'<text x="{W-PAD-90:.0f}" y="{ly+18:.0f}" fill="#333">dashed=RAW</text>')
    s.append("</svg>")
    return "\n".join(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="a calib dir, or gamma.csv [raw_gamma.csv]")
    ap.add_argument("-o", "--out", default=None, help="output SVG (default <dir>/gamma.svg)")
    args = ap.parse_args()

    isp_path = raw_path = None
    outdir = "."
    for p in args.inputs:
        if os.path.isdir(p):
            outdir = p
            for name, set_isp in (("gamma.csv", True), ("raw_gamma.csv", False)):
                fp = os.path.join(p, name)
                if os.path.exists(fp):
                    if set_isp:
                        isp_path = fp
                    else:
                        raw_path = fp
        elif p.endswith("raw_gamma.csv"):
            raw_path = p
        else:
            isp_path = p
            outdir = os.path.dirname(p) or "."

    if not isp_path and not raw_path:
        sys.exit("ERROR: no gamma.csv / raw_gamma.csv found")

    isp = load(isp_path) if isp_path else None
    raw = load(raw_path) if raw_path else None
    print(f"ISP: {isp_path or '-'}   RAW: {raw_path or '-'}")

    svg = build_svg(isp, raw)
    out = args.out or os.path.join(outdir, "gamma.svg")
    with open(out, "w") as f:
        f.write(svg)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
