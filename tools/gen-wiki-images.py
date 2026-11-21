#!/usr/bin/env python3
"""Illustrations for the hyperion-cam wiki (Hyperion-Cam-Setup.md). Drawn, not photographed.

Usage: python3 tools/gen-wiki-images.py <wiki-checkout>/images [overview crop transform rois flow webstart eyecheck]
Needs Pillow, numpy, opencv-python and the DejaVu fonts. With no names it draws all images.
"""
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1] if len(sys.argv) > 1 else 'images'
os.makedirs(OUT, exist_ok=True)
S = 2  # supersampling

FONTS = {'r': '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
         'b': '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
         'm': '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
         'mb': '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'}
_fc = {}


def font(sz, w='r'):
    k = (sz, w)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(FONTS[w], round(sz * S))
    return _fc[k]


INK = (31, 35, 40)
MUTED = (89, 99, 110)
BG = (255, 255, 255)
BLUE = (9, 105, 218)
ORANGE = (204, 102, 24)
GREEN = (26, 127, 55)
GRAY = (110, 119, 129)
WALL = (36, 36, 42)
CROPG = (90, 225, 130)
YEL = (245, 205, 60)
RED = (240, 70, 70)
CYAN = (70, 215, 235)


class Canvas:
    def __init__(self, w, h, bg=BG):
        self.w, self.h = w, h
        self.im = Image.new('RGB', (w * S, h * S), bg)
        self.d = ImageDraw.Draw(self.im)

    @staticmethod
    def p(*v):
        return [round(x * S) for x in v]

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, width=1.0, r=0):
        box = self.p(x0, y0, x1, y1)
        wd = max(1, round(width * S)) if outline else 0
        if r:
            self.d.rounded_rectangle(box, radius=round(r * S), fill=fill, outline=outline, width=wd)
        else:
            self.d.rectangle(box, fill=fill, outline=outline, width=wd)

    def line(self, pts, fill, width=1.0):
        self.d.line([tuple(self.p(*q)) for q in pts], fill=fill, width=max(1, round(width * S)), joint='curve')

    def dline(self, p0, p1, fill, width=1.5, dash=8, gap=6):
        x0, y0 = p0
        x1, y1 = p1
        L = math.hypot(x1 - x0, y1 - y0)
        if L == 0:
            return
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        t = 0.0
        while t < L:
            t1 = min(L, t + dash)
            self.line([(x0 + ux * t, y0 + uy * t), (x0 + ux * t1, y0 + uy * t1)], fill, width)
            t = t1 + gap

    def drect(self, x0, y0, x1, y1, fill, width=1.5, dash=8, gap=6):
        for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
            self.dline(a, b, fill, width, dash, gap)

    def head(self, tip, frm, fill, size=11):
        ang = math.atan2(tip[1] - frm[1], tip[0] - frm[0])
        a1, a2 = ang + math.radians(150), ang - math.radians(150)
        pts = [tip, (tip[0] + size * math.cos(a1), tip[1] + size * math.sin(a1)),
               (tip[0] + size * math.cos(a2), tip[1] + size * math.sin(a2))]
        self.d.polygon([tuple(self.p(*q)) for q in pts], fill=fill)

    def arrow(self, pts, fill, width=2.5, head=11, both=False):
        # stop the shaft short of the tip so the head stays sharp
        (ax, ay), (bx, by) = pts[-2], pts[-1]
        L = math.hypot(bx - ax, by - ay)
        shaft = list(pts[:-1]) + [(bx - (bx - ax) / L * head * 0.6, by - (by - ay) / L * head * 0.6)]
        if both:
            (cx, cy), (dx, dy) = pts[1], pts[0]
            L2 = math.hypot(dx - cx, dy - cy)
            shaft[0] = (dx - (dx - cx) / L2 * head * 0.6, dy - (dy - cy) / L2 * head * 0.6)
            self.head(pts[0], pts[1], fill, head)
        self.line(shaft, fill, width)
        self.head(pts[-1], pts[-2], fill, head)

    def text(self, x, y, t, sz=15, w='r', fill=INK, anchor='la'):
        self.d.text((x * S, y * S), t, font=font(sz, w), fill=fill, anchor=anchor)

    def tw(self, t, sz=15, w='r'):
        return self.d.textlength(t, font=font(sz, w)) / S

    def circle(self, x, y, r, fill, outline=None, width=1.0):
        self.d.ellipse(self.p(x - r, y - r, x + r, y + r), fill=fill, outline=outline,
                       width=max(1, round(width * S)) if outline else 0)

    def paste(self, arr, x, y):
        self.im.paste(Image.fromarray(arr), (round(x * S), round(y * S)))

    def save(self, name):
        self.im.resize((self.w, self.h), Image.LANCZOS).save(os.path.join(OUT, name), optimize=True)
        print('wrote', os.path.join(OUT, name))


# ---------------------------------------------------------------- scene pieces (S-scaled arrays)

def landscape(w, h):
    """A simple 'film frame': sky, sun, two hills."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    v = yy / h
    u = xx / w
    top = np.array([58, 118, 222], np.float32)
    hor = np.array([178, 212, 246], np.float32)
    img = top + (hor - top) * (v[..., None] ** 0.8)
    cx, cy, r = 0.26 * w, 0.32 * h, 0.085 * h
    dd = np.hypot(xx - cx, yy - cy)
    halo = np.clip(1 - (dd - r) / (r * 2.4), 0, 1) ** 2
    img = img * (1 - halo[..., None] * 0.55) + np.array([255, 222, 150], np.float32) * halo[..., None] * 0.55
    img[dd < r] = [255, 196, 70]
    far = v > 0.60 + 0.05 * np.sin(u * 7.0 + 0.5) + 0.025 * np.sin(u * 19.0)
    img[far] = (np.array([84, 150, 104], np.float32) * (1.0 - 0.25 * (v[far, None] - 0.55)))
    near = v > 0.74 + 0.06 * np.sin(u * 4.3 + 2.0)
    img[near] = (np.array([52, 128, 56], np.float32) * (1.05 - 0.45 * (v[near, None] - 0.7)))
    return img.clip(0, 255).astype(np.uint8)


def calib_page(w, h):
    """The calibration page start screen as the camera sees it (no text)."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (17, 17, 17)
    f = math.sqrt(0.10)
    bw, bh = round(w * f), round(h * f)
    x0, y0 = (w - bw) // 2, (h - bh) // 2
    t = max(3, round(h * 4 / 1080 * 3.0))
    img[y0:y0 + bh, x0:x0 + bw] = (226, 34, 34)
    img[y0 + t:y0 + bh - t, x0 + t:x0 + bw - t] = (85, 85, 85)
    return img


def solid(w, h, rgb):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = rgb
    return img


def scene(w, h, quad, screen, wall=WALL, glow_rgb=None, glow_gain=0.9, bezel=5.0, glow_sigma=0.07):
    """Wall + LED glow + TV. quad = TL, TR, BL, BR of the picture area, in canvas units."""
    W, H = round(w * S), round(h * S)
    q = np.array(quad, np.float64) * S
    out = np.zeros((H, W, 3), np.float32) + np.array(wall, np.float32)
    sh, sw = screen.shape[:2]

    def qpt(u, v):
        return (1 - u) * (1 - v) * q[0] + u * (1 - v) * q[1] + (1 - u) * v * q[2] + u * v * q[3]

    if glow_gain > 0:
        layer = np.zeros((H, W, 3), np.float32)
        N = 10
        T = max(8, int(0.10 * min(W, H)))
        for name in ('top', 'bottom', 'left', 'right'):
            for k in range(N):
                t0, t1 = k / N, (k + 1) / N
                if glow_rgb is not None:
                    col = np.array(glow_rgb, np.float32)
                elif name == 'top':
                    col = screen[:int(sh * .15), int(sw * t0):int(sw * t1)].reshape(-1, 3).mean(0)
                elif name == 'bottom':
                    col = screen[int(sh * .85):, int(sw * t0):int(sw * t1)].reshape(-1, 3).mean(0)
                elif name == 'left':
                    col = screen[int(sh * t0):int(sh * t1), :int(sw * .12)].reshape(-1, 3).mean(0)
                else:
                    col = screen[int(sh * t0):int(sh * t1), int(sw * .88):].reshape(-1, 3).mean(0)
                a, b = {'top': ((t0, 0), (t1, 0)), 'bottom': ((t0, 1), (t1, 1)),
                        'left': ((0, t0), (0, t1)), 'right': ((1, t0), (1, t1))}[name]
                p0, p1 = qpt(*a), qpt(*b)
                cv2.line(layer, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                         tuple(float(c) for c in col), T)
        layer = cv2.GaussianBlur(layer, (0, 0), glow_sigma * min(W, H))
        out += layer * glow_gain
    c = q.mean(0)
    qb = [p + np.sign(p - c) * bezel * S for p in q]
    cv2.fillPoly(out, [np.array([qb[0], qb[1], qb[3], qb[2]], np.int32)], (16.0, 16.0, 18.0), cv2.LINE_AA)
    src = np.float32([[0, 0], [sw, 0], [0, sh], [sw, sh]])
    M = cv2.getPerspectiveTransform(src, np.float32(q))
    warped = cv2.warpPerspective(screen.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones((sh, sw), np.float32), M, (W, H), flags=cv2.INTER_LINEAR)
    out = out * (1 - mask[..., None]) + warped * mask[..., None]
    return out.clip(0, 255).astype(np.uint8)


def rounded_paste(cv, arr, x, y, w, h, r=12):
    """Paste a scene array with rounded corners."""
    img = Image.fromarray(arr)
    m = Image.new('L', img.size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, img.size[0] - 1, img.size[1] - 1], radius=r * S, fill=255)
    cv.im.paste(img, (round(x * S), round(y * S)), m)


def camera_icon(cv, x, y, s=1.0):
    """Small camera board + lens, centred on x,y."""
    cv.rect(x - 30 * s, y - 20 * s, x + 30 * s, y + 20 * s, fill=(28, 120, 70), outline=(150, 220, 170), width=1.5, r=4)
    cv.circle(x, y, 11 * s, (20, 20, 22), outline=(200, 200, 205), width=1.5)
    cv.circle(x, y, 5 * s, (60, 70, 110))
    cv.rect(x - 6 * s, y + 20 * s, x + 6 * s, y + 34 * s, fill=(70, 70, 76))


def box(cv, x, y, w, h, title, lines, outline=(208, 215, 222), fill=(246, 248, 250), owidth=1.5):
    cv.rect(x, y, x + w, y + h, fill=fill, outline=outline, width=owidth, r=10)
    cv.text(x + w / 2, y + 22, title, 17, 'b', INK, 'mm')
    for i, (t, st) in enumerate(lines):
        cv.text(x + w / 2, y + 50 + i * 21, t, 13, st, MUTED, 'mm')


# ---------------------------------------------------------------- 1. overview

def overview():
    cv = Canvas(1200, 640)
    pw, ph = 1120, 410
    tv = [(310, 40), (810, 40), (310, 321), (810, 321)]
    arr = scene(pw, ph, tv, landscape(round(500 * S), round(281 * S)), glow_gain=1.0)
    rounded_paste(cv, arr, 40, 20, pw, ph, 14)
    camera_icon(cv, 600, 380)
    cv.text(648, 380, 'camera, aimed at the TV', 15, 'r', (215, 215, 220), 'lm')
    cv.text(70, 52, 'LED strip on the back', 15, 'r', (215, 215, 220))
    cv.text(70, 72, 'of the TV lights the wall', 15, 'r', (215, 215, 220))

    bx = [40, 340, 640, 940]
    items = [('Camera', [('Raspberry Pi camera', 'r'), ('(libcamera)', 'r')]),
             ('hyperion-cam', [('crop · perspective warp', 'r'), ('shading · colour LUT', 'r')]),
             ('HyperHDR', [('LED areas · smoothing', 'r'), ('white balance · gamma', 'r')]),
             ('LED strip', [('e.g. WS2812 / SK6812', 'r'), ('', 'r')])]
    for i, (x, (t, ls)) in enumerate(zip(bx, items)):
        if i == 1:
            box(cv, x, 465, 220, 110, t, ls, outline=BLUE, fill=(240, 246, 255), owidth=2.5)
        else:
            box(cv, x, 465, 220, 110, t, ls)
        if i < 3:
            cv.arrow([(x + 226, 520), (bx[i + 1] - 6, 520)], GRAY, 3, 12)
    cv.text(600, 608, 'hyperion-cam sends the frames to HyperHDR over a local Unix socket (flatbuffer protocol).',
            14, 'r', MUTED, 'mm')
    cv.save('overview.png')


# ---------------------------------------------------------------- 2. crop

def crop():
    cv = Canvas(1200, 650)
    fx, fy, fw, fh = 120, 52, 960, 540
    tv = [(350, 190), (670, 182), (354, 370), (666, 378)]
    arr = scene(fw, fh, tv, landscape(round(320 * S), round(190 * S)), glow_gain=1.0)
    cv.paste(arr, fx, fy)
    cv.rect(fx, fy, fx + fw, fy + fh, outline=(150, 150, 158), width=1.5)
    cv.text(fx, fy - 12, 'Full camera sensor (4608 × 2592 on Camera Module 3). --crop is given in these sensor pixels.',
            15, 'r', INK, 'lb')

    def L(x, y):
        return fx + x, fy + y
    # area used at -w/-h (central 2/3 on imx708 for the 1536x864 mode)
    x0, y0 = L(160, 90)
    x1, y1 = L(800, 450)
    cv.rect(x0, y0, x1, y1, outline=RED, width=2.5)
    cv.text(x0, y1 + 10, 'red: area the camera uses at -w 768 -h 432 (drawn by --imgout-crophelp)', 14, 'r', RED, 'lt')
    # the user's crop
    cx0, cy0 = L(328, 160)
    cx1, cy1 = L(692, 402)
    cv.rect(cx0, cy0, cx1, cy1, outline=CROPG, width=3)
    cv.circle(cx0, cy0, 6, CROPG)
    cv.text(cx0 - 10, cy0 - 8, '(x, y)', 15, 'mb', CROPG, 'rb')
    cv.arrow([(cx0 + 2, cy0 - 18), (cx1 - 2, cy0 - 18)], CROPG, 2, 9, both=True)
    cv.text((cx0 + cx1) / 2, cy0 - 24, 'width', 14, 'm', CROPG, 'mb')
    cv.arrow([(cx1 + 18, cy0 + 2), (cx1 + 18, cy1 - 2)], CROPG, 2, 9, both=True)
    cv.text(cx1 + 28, (cy0 + cy1) / 2, 'height', 14, 'm', CROPG, 'lm')
    cv.text((cx0 + cx1) / 2, cy1 + 14, '--crop x,y,width,height', 16, 'mb', CROPG, 'mt')
    cv.text(600, 622, 'Frame the TV picture with a small margin and keep the crop inside the red rectangle.',
            14, 'r', MUTED, 'mm')
    cv.save('crop.png')


# ---------------------------------------------------------------- 3. transform

def corner_dot(cv, x, y, n):
    cv.circle(x, y, 12, (225, 45, 45), outline=(255, 255, 255), width=2)
    cv.text(x, y + 0.5, str(n), 13, 'b', (255, 255, 255), 'mm')


def tag(cv, x, y, t, anchor='la'):
    w = cv.tw(t, 13, 'm')
    h = 20
    ox = {'l': 0, 'r': -w - 12}[anchor[0]]
    oy = {'t': 0, 'b': -h}[anchor[1]]
    cv.rect(x + ox, y + oy, x + ox + w + 12, y + oy + h, fill=(20, 20, 24), r=4)
    cv.text(x + ox + 6, y + oy + h / 2, t, 13, 'm', (240, 240, 240), 'lm')


def transform():
    cv = Canvas(1200, 470)
    lx, ly, pw, ph = 40, 60, 512, 288
    tv = [(26, 22), (490, 8), (20, 272), (498, 282)]
    arr = scene(pw, ph, tv, landscape(round(480 * S), round(270 * S)), glow_gain=0.8, bezel=4)
    rounded_paste(cv, arr, lx, ly, pw, ph, 8)
    cv.text(lx, ly - 12, 'Camera picture with --crop (from --imgout)', 15, 'b', INK, 'lb')
    labels = [('x1,y1', 'lt'), ('x2,y2', 'rt'), ('x3,y3', 'lb'), ('x4,y4', 'rb')]
    for i, ((x, y), (t, an)) in enumerate(zip(tv, labels)):
        X, Y = lx + x, ly + y
        corner_dot(cv, X, Y, i + 1)
        tag(cv, X + (18 if an[0] == 'l' else -18), Y + (14 if an[1] == 't' else -14), t, an)
    rx = 648
    rounded_paste(cv, landscape(round(pw * S), round(ph * S)), rx, ly, pw, ph, 8)
    cv.text(rx, ly - 12, 'Output after -t (sent to HyperHDR)', 15, 'b', INK, 'lb')
    for i, (x, y) in enumerate([(14, 14), (pw - 14, 14), (14, ph - 14), (pw - 14, ph - 14)]):
        corner_dot(cv, rx + x, ly + y, i + 1)
    cv.arrow([(566, 204), (636, 204)], BLUE, 4, 14)
    cv.text(600, 192, '-t', 16, 'mb', BLUE, 'mb')
    cv.text(600, 382, '-t x1,y1:x2,y2:x3,y3:x4,y4', 18, 'mb', INK, 'mm')
    cv.text(600, 412, 'Order: 1 top-left, 2 top-right, 3 bottom-left, 4 bottom-right, in pixels of the --imgout picture.',
            14, 'r', MUTED, 'mm')
    cv.text(600, 436, 'Put the points on the corners of the TV picture, not on the bezel.', 14, 'r', MUTED, 'mm')
    cv.save('transform.png')


# ---------------------------------------------------------------- 4. ROIs

def rois():
    cv = Canvas(1200, 480)
    pw, ph = 552, 310
    ax, bx, py = 32, 616, 60
    # (a) TV ROI on the cropped, un-warped view of the start page
    tv = [(24, 20), (530, 8), (18, 296), (536, 302)]
    arr = scene(pw, ph, tv, calib_page(round(512 * S), round(288 * S)), glow_gain=0.0, bezel=4)
    rounded_paste(cv, arr, ax, py, pw, ph, 8)
    q = np.array(tv, float)

    def qpt(u, v):
        return (1 - u) * (1 - v) * q[0] + u * (1 - v) * q[1] + (1 - u) * v * q[2] + u * v * q[3]
    p0, p1 = qpt(0.395, 0.40), qpt(0.605, 0.60)
    cv.drect(ax + p0[0], py + p0[1], ax + p1[0], py + p1[1], CYAN, 2.5, 7, 5)
    cv.text(ax, py - 12, 'TV ROI: cropped view (-w/-h + --crop, no -t)', 15, 'b', INK, 'lb')
    cv.text(ax + pw / 2, py + ph + 22, '--calib-tv-roi x,y,w,h', 15, 'mb', BLUE, 'mm')
    cv.text(ax + pw / 2, py + ph + 46, 'Inside the red rectangle of the start page,', 14, 'r', MUTED, 'mm')
    cv.text(ax + pw / 2, py + ph + 66, 'away from its border.', 14, 'r', MUTED, 'mm')
    # (b) LED ROI on the full frame, LEDs lit white, TV black
    tv2 = [(166, 92), (386, 88), (168, 212), (384, 216)]
    arr = scene(pw, ph, tv2, solid(100, 56, (8, 8, 9)), glow_rgb=(255, 236, 212), glow_gain=0.85,
                bezel=3, glow_sigma=0.09)
    rounded_paste(cv, arr, bx, py, pw, ph, 8)
    r0x, r0y, r1x, r1y = bx + 398, py + 120, bx + 414, py + 182
    cv.rect(r0x, r0y, r1x, r1y, outline=CYAN, width=2.5)
    cv.line([(r1x + 4, (r0y + r1y) / 2), (r1x + 46, (r0y + r1y) / 2)], CYAN, 1.5)
    cv.text(r1x + 50, (r0y + r1y) / 2, 'LED ROI', 14, 'b', CYAN, 'lm')
    cv.text(bx, py - 12, 'LED ROI: full frame (-w 1024 -h 576, no --crop)', 15, 'b', INK, 'lb')
    cv.text(bx + pw / 2, py + ph + 22, '--calib-led-roi x,y,w,h', 15, 'mb', BLUE, 'mm')
    cv.text(bx + pw / 2, py + ph + 46, 'On evenly lit wall next to the TV,', 14, 'r', MUTED, 'mm')
    cv.text(bx + pw / 2, py + ph + 66, 'not on the TV picture.', 14, 'r', MUTED, 'mm')
    cv.save('rois.png')


# ---------------------------------------------------------------- 5. calibration flow

def flow():
    cv = Canvas(1200, 540)
    cats = {'tv': (BLUE, 'camera measures the TV (web page)'),
            'led': (ORANGE, 'camera measures the LED glow'),
            'eye': (GREEN, 'you compare by eye'),
            'off': (GRAY, 'offline, no camera')}
    x = 40
    for k in ('tv', 'led', 'eye', 'off'):
        col, t = cats[k]
        cv.rect(x, 18, x + 16, 34, fill=col, r=3)
        cv.text(x + 24, 26, t, 14, 'r', INK, 'lm')
        x += 24 + cv.tw(t, 14) + 36
    steps = [
        (0, 'White balance', 'eye', ['HyperHDR temperature', '~5 min']),
        (1, 'TV RAW probe', 'tv', ['tv_raw.csv', '~2 min']),
        (2, 'Frozen tuning', 'off', ['tuning.json', 'seconds']),
        (3, 'Shading', 'tv', ['shading.csv', '~2 min · optional']),
        (4, 'Camera response', 'led', ['camera_response.csv', '~1 min']),
        (5, 'TV sweep', 'tv', ['tv_sweep.csv', '~1 h']),
        (6, 'LED sweep', 'led', ['led_sweep.csv', '~45 min']),
        (7, 'Build the LUT', 'off', ['lut.cube', '~1 min']),
        (8, 'Eye check', 'eye', ['slider page', 'a few minutes']),
        (9, 'Apply', 'off', ['hypercam service', 'restart']),
    ]
    W, H, G = 200, 132, 30
    rows = [62, 292]
    for i, (n, t, c, ls) in enumerate(steps):
        col = cats[c][0]
        x = 40 + (i % 5) * (W + G)
        y = rows[i // 5]
        cv.rect(x, y, x + W, y + H, fill=(255, 255, 255), outline=col, width=2, r=10)
        cv.d.rounded_rectangle(cv.p(x, y, x + W, y + 36), radius=10 * S, fill=col, corners=(True, True, False, False))
        cv.text(x + 12, y + 18, str(n), 17, 'b', (255, 255, 255), 'lm')
        cv.text(x + 34, y + 18, t, 15, 'b', (255, 255, 255), 'lm')
        cv.text(x + W / 2, y + 66, ls[0], 13, 'm', INK, 'mm')
        cv.text(x + W / 2, y + 96, ls[1], 13, 'r', MUTED, 'mm')
        if i % 5 < 4:
            cv.arrow([(x + W + 3, y + H / 2), (x + W + G - 3, y + H / 2)], GRAY, 2.5, 10)
    x4 = 40 + 4 * (W + G) + W / 2
    x5 = 40 + W / 2
    cv.arrow([(x4, rows[0] + H + 3), (x4, 248), (x5, 248), (x5, rows[1] - 3)], GRAY, 2.5, 10)
    cv.text(600, 466, 'Use one new output folder for every step. All camera steps use the same camera settings,',
            14, 'r', MUTED, 'mm')
    cv.text(600, 488, 'and from step 2 on the same tuning file. Times are from a real run (TV sweep at grid 16).',
            14, 'r', MUTED, 'mm')
    cv.save('calib-flow.png')


# ---------------------------------------------------------------- 6. web page start screen

def tv_frame(cv, sx0, sy0, sx1, sy1):
    cv.rect(sx0 - 12, sy0 - 12, sx1 + 12, sy1 + 12, fill=(12, 12, 14), r=6)
    cx = (sx0 + sx1) / 2
    cv.rect(cx - 40, sy1 + 12, cx + 40, sy1 + 40, fill=(22, 22, 25))
    cv.rect(cx - 120, sy1 + 40, cx + 120, sy1 + 50, fill=(22, 22, 25), r=3)


def webstart():
    cv = Canvas(1200, 700, bg=(30, 30, 34))
    sx0, sy0, sx1, sy1 = 100, 60, 1100, 622
    tv_frame(cv, sx0, sy0, sx1, sy1)
    cv.rect(sx0, sy0, sx1, sy1, fill=(17, 17, 17))
    W, H = sx1 - sx0, sy1 - sy0
    f = math.sqrt(0.10)
    bw, bh = W * f, H * f
    px0, py0 = sx0 + (W - bw) / 2, sy0 + (H - bh) / 2
    cv.rect(px0, py0, px0 + bw, py0 + bh, fill=(85, 85, 85), outline=(226, 34, 34), width=3)
    cv.text(600, sy0 + 26, 'Tap / click to begin (goes fullscreen)', 21, 'r', (204, 204, 204), 'mt')
    cv.text(600, sy0 + 60,
            'patch 10% of screen, 607×342 px — screenshot it now (run --imgout) and set --calib-tv-roi inside the red rectangle.',
            12.5, 'r', (187, 187, 187), 'mt')
    cv.text(sx0 + 8, sy1 - 8, 'connected', 12, 'r', (95, 95, 95), 'lb')
    cv.save('web-start.png')


# ---------------------------------------------------------------- 7. eye check page

def button(cv, x, y, t, on=False, sz=13):
    w = cv.tw(t, sz) + 16
    cv.rect(x, y - 12, x + w, y + 12, fill=(34, 170, 119) if on else (233, 233, 237),
            outline=None if on else (150, 150, 150), width=1, r=3)
    cv.text(x + w / 2, y, t, sz, 'r', (255, 255, 255) if on else (30, 30, 30), 'mm')
    return x + w + 6


def eyecheck():
    cv = Canvas(1200, 640, bg=(30, 30, 34))
    sx0, sy0, sx1, sy1 = 170, 60, 1030, 544
    tv = [(sx0, sy0), (sx1, sy0), (sx0, sy1), (sx1, sy1)]
    arr = scene(1200, 640, tv, solid(200, 112, (127, 127, 127)), wall=(30, 30, 34),
                glow_rgb=(172, 166, 158), glow_gain=0.95, bezel=12, glow_sigma=0.06)
    cv.paste(arr, 0, 0)
    cx = 600
    cv.rect(cx - 40, sy1 + 12, cx + 40, sy1 + 40, fill=(22, 22, 25))
    cv.rect(cx - 120, sy1 + 40, cx + 120, sy1 + 50, fill=(22, 22, 25), r=3)
    p0x, p1x, p0y, p1y = 245, 955, 366, 526
    cv.rect(p0x, p0y, p1x, p1y, fill=(51, 51, 51), r=6)
    y = p0y + 24
    cv.text(p0x + 16, y, 'hyperion-cam eye check', 14, 'b', (238, 238, 238), 'lm')
    cv.text(p0x + 16 + cv.tw('hyperion-cam eye check', 14, 'b'), y, ' - compare the LED glow with the TV', 14, 'r',
            (238, 238, 238), 'lm')
    bxx = p1x - 16 - (cv.tw('gray', 13) + cv.tw('colour', 13) + cv.tw('fullscreen', 13) + 3 * 16 + 12)
    bxx = button(cv, bxx, y, 'gray', True)
    bxx = button(cv, bxx, y, 'colour')
    button(cv, bxx, y, 'fullscreen')
    y += 36
    cv.text(p0x + 16, y, 'brightness', 14, 'r', (238, 238, 238), 'lm')
    t0, t1 = p0x + 130, p1x - 70
    cv.rect(t0, y - 3, t1, y + 3, fill=(205, 205, 205), r=3)
    kx = t0 + (t1 - t0) * 127 / 255
    cv.rect(t0, y - 3, kx, y + 3, fill=(0, 117, 255), r=3)
    cv.circle(kx, y, 9, (0, 117, 255), outline=(255, 255, 255), width=2)
    cv.text(p1x - 16, y, '127', 14, 'r', (238, 238, 238), 'rm')
    y += 34
    bxx = p0x + 16
    for v in (0, 8, 16, 32, 48, 64, 96, 127, 160, 192, 224, 255):
        bxx = button(cv, bxx, y, str(v), sz=12)
    y += 36
    cv.text(p0x + 16, y, 'RGB 127, 127, 127', 14, 'r', (238, 238, 238), 'lm')
    cv.text(p1x - 16, y, '←/→ ±1   ↓/↑ ±16   g/c mode   f fullscreen   h hide', 12, 'r', (170, 170, 170), 'rm')
    cv.save('eyecheck.png')


if __name__ == '__main__' and len(sys.argv) > 2:
    for n in sys.argv[2:]:
        globals()[n]()
elif __name__ == '__main__':
    overview()
    crop()
    transform()
    rois()
    flow()
    webstart()
    eyecheck()
