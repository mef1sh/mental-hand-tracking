# M.E.N.T.A.L. - version 4.1 (easier pinch: thumb touching a finger is detected much more reliably)
#
# Control your computer with your hands through a camera.
#
#   CURSOR       : moves ONLY when you show one index finger (the other fingers are bent). Any hand.
#   RIGHT hand   : thumb + index  = left mouse button   (hold the pinch = hold the button, you can drag)
#                  thumb + middle = right mouse button
#   LEFT hand    : thumb + index  = VOLUME  (move the hand up / right = louder, down / left = quieter)
#                  thumb + middle = SCROLL  (hand down = page goes up, hand up = page goes down)
#   Both hands work at the same time: for example the left hand points and moves the cursor,
#   the right hand holds the button (drag and drop).
#   WIN + TAB    : both hands from fists to open palms = open Win+Tab
#                  both hands back to fists            = close it (Esc)
#                  inside Win+Tab: ONE open hand moves right = arrow RIGHT, moves left = arrow LEFT
#                                  ONE hand makes a fist     = Enter (open the chosen window)
#
# Keys in the camera window:  q = exit   c = next camera   h = swap left/right hands   d = debug numbers

import os                       # files and folders
import sys                      # command line
import math                     # math (distances)
import time                     # time
import argparse                 # command line options (--camera 1)
import threading                # threads: doing several things at once
import traceback                # to print errors nicely
import urllib.request           # downloading the model file
from collections import deque   # a list that is quick to trim
import cv2                      # OpenCV: camera and drawing
import mediapipe as mp          # the neural network that finds the hand
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------- 0. Stop Windows from slowing the program down when the window is minimized ----------
if os.name == "nt":
    try:
        import ctypes
        from ctypes import wintypes

        class PowerThrottling(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG),
                        ("ControlMask", wintypes.ULONG),
                        ("StateMask", wintypes.ULONG)]

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                   ctypes.c_void_p, wintypes.DWORD]
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]

        me = kernel32.GetCurrentProcess()
        state = PowerThrottling(1, 5, 0)     # 0 = don't throttle anything
        kernel32.SetProcessInformation(me, 4, ctypes.byref(state), ctypes.sizeof(state))
        kernel32.SetPriorityClass(me, 0x00008000)   # slightly higher priority
        ctypes.windll.winmm.timeBeginPeriod(1)      # precise timers
    except Exception as e:
        print("Could not disable power saving:", e)

# ---------- 1. Neural network model (downloaded once) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

# ---------- 2. Settings (you can tweak everything here) ----------
# --- camera ---
CAMERA_NUMBER = 0        # camera number (0 = first, 1 = second, e.g. Camo). Also: python M.E.N.T.A.L..py --camera 1
MAX_CAMERAS = 5          # how many camera numbers the "c" key will try
CAM_WIDTH = 640          # camera frame width
CAM_HEIGHT = 480         # camera frame height
CAM_FPS = 60             # ask the camera for 60 frames per second (if it can)
DETECT_WIDTH = 480       # the neural network looks at a shrunk picture (bigger = fingers are more precise, a bit slower)
SHOW_WINDOW = True       # show the camera window
PREVIEW_ON_TOP = True    # keep the camera window above other windows
SWAP_HANDS = True        # the picture is a mirror, so the hands are swapped by default (you can also press "h" in the window)

# --- cursor ---
ACTIVE_X = (0.12, 0.88)  # the part of the camera picture (left..right) that covers the whole screen width
ACTIVE_Y = (0.10, 0.78)  # the part of the camera picture (top..bottom) that covers the whole screen height
CURSOR_MIN_CUTOFF = 1.2  # smoothing when the finger is still (smaller = steadier, but slower)
CURSOR_BETA = 0.03       # smoothing when the finger is fast (bigger = less lag on fast moves)
CURSOR_REWIND = 0.12     # when you start a pinch the cursor goes back this many seconds (the fingers move while closing)
CURSOR_CATCHUP = 0.12    # after a pinch the cursor glides back to the fingertip in about this many seconds
DRAG_DEADZONE_PX = 18    # after a button press the cursor stays still until the hand moves this many pixels

# --- pinch (thumb touching a finger). Distances are in "hand sizes", so they work near and far from the camera ---
PINCH_ON = 0.42          # thumb-finger distance smaller than this = pinch (bigger number = easier to click)
PINCH_OFF = 0.60         # bigger than this = pinch is released
APPROACH = 0.85          # fingers closer than this = "about to pinch": the cursor freezes (so a click doesn't miss)
OTHER_RATIO = 0.8        # thumb+index and thumb+middle must not be both close: the chosen finger must be this much closer
PINCH_MIN_EXT = 1.15     # the pinching finger must not be curled into the palm (that is a fist, no clicks by accident)
FIST_THUMB_GAP = 0.40    # a hand with the thumb touching a finger tip is a pinch, not a fist...
FIST_TIGHT = 1.15        # ...unless every finger is really curled into the palm
PRESS_DELAY = 0.03       # how long the pinch must hold before it counts
RELEASE_DELAY = 0.04     # how long the fingers must stay apart before the release counts

# --- finger straight / bent (finger_length_from_wrist / knuckle_length_from_wrist) ---
EXT_DOWN = 1.35          # smaller = finger is bent (down)
EXT_OPEN = 1.50          # all 4 fingers bigger than this = open palm
PALM_EXT = 1.45          # at least 3 fingers bigger than this = "a whole hand is shown" (for the Win+Tab arrows)

# --- the "pointing" pose: only the index finger is straight, the others are bent ---
POINT_START_IDX = 1.55   # index must be straighter than this to start pointing
POINT_START_OTHER = 1.35 # the other three fingers must be more bent than this
POINT_STOP_IDX = 1.40    # pointing stops if the index gets more bent than this...
POINT_STOP_OTHER = 1.50  # ...or any other finger gets straighter than this

# --- fists / open palms / Win+Tab ---
POSE_HOLD_OPEN = 0.10    # how long the open palms must be held
POSE_HOLD_FIST = 0.15    # how long the fists must be held
SYNC_WINDOW = 0.6        # both hands must change within this many seconds ("at the same time")
FIST_MEMORY = 0.8        # the palms must be opened this soon after the fists
FIST_LOCKOUT = 0.45      # after a fist this long there are no clicks (opening a fist must not click)
TOGGLE_PAUSE = 1.0       # pause after opening/closing Win+Tab
TASKVIEW_CLOSE = "esc"   # how two fists close Win+Tab: "esc" = press Esc, "wintab" = press Win+Tab again
CLOSE_SYNC = 0.4         # two fists must appear within this many seconds to close Win+Tab
ENTER_HOLD = 0.4         # ONE fist must be held this long to press Enter (so two fists don't press it by accident)
SWIPE_STEP = 0.07        # the hand must get this far (share of the picture) ahead of its "start point" to press one arrow
SWIPE_LEAK = 3.0         # the start point catches up with the hand: so only a FAST movement presses an arrow,
                         # a slow movement (e.g. taking the hand back) does nothing. Smaller = slow movements work too
SWIPE_COOLDOWN = 0.35    # minimum pause between two arrow presses

# --- volume and scroll (hand moves away from the point where you pinched, like a joystick) ---
MODE_DEADZONE = 0.03     # the hand must move this much (share of the picture) before anything happens
PALM_SMOOTH = 0.5        # hand smoothing: smaller = smoother
VOLUME_SPEED = 120       # volume steps per second for every 1.0 of hand shift (bigger = faster)
VOLUME_MAX = 20          # maximum volume steps per second (1 step = 2%)
SCROLL_SPEED = 150       # scroll "clicks of the wheel" per second for every 1.0 of hand shift
SCROLL_MAX = 40          # maximum wheel clicks per second

# --- tracking ---
GRACE = 0.25             # if the camera loses the hand for less than this, nothing is released
MATCH_DIST = 0.30        # how far a hand may jump between frames and still be the same hand

# --- colors (B, G, R) - every action has its own color ---
COLORS = {
    "idle":   (170, 170, 170),   # grey
    "cursor": (255, 170, 0),     # blue
    "left":   (0, 220, 0),       # green
    "right":  (0, 0, 255),       # red
    "volume": (0, 150, 255),     # orange
    "scroll": (255, 0, 170),     # purple
    "wintab": (0, 255, 255),     # yellow
}
NAMES = [("cursor", "CURSOR"), ("left", "LEFT CLICK"), ("right", "RIGHT CLICK"),
         ("volume", "VOLUME"), ("scroll", "SCROLL"), ("wintab", "WIN+TAB")]
PRIORITY = ["volume", "scroll", "right", "left", "wintab", "cursor"]   # which color paints the frame border

FINGERTIPS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky


# ---------- 3. Small helpers ----------
def clamp(value, low, high):
    """Keeps the number inside the limits low..high."""
    return max(low, min(high, value))


def softdz(v, dz):
    """Soft dead zone: inside +-dz it's 0, outside it grows smoothly from 0 (no sudden jump)."""
    if v > dz:
        return v - dz
    if v < -dz:
        return v + dz
    return 0.0


class OneEuro:
    """Smart smoothing: strong when the finger is still, light when it moves fast."""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.t = None
        self.x = 0.0
        self.dx = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t is None:
            self.t, self.x, self.dx = t, x, 0.0
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


# ---------- 4. Talking to Windows: mouse and keyboard ----------
class WinActuator:
    """Moves the mouse, presses buttons and keys using Windows functions (fast, no pauses)."""
    VK_TAB, VK_RETURN, VK_ESCAPE, VK_LWIN = 0x09, 0x0D, 0x1B, 0x5B
    VK_LEFT, VK_RIGHT = 0x25, 0x27
    VK_VOL_DOWN, VK_VOL_UP = 0xAE, 0xAF

    def __init__(self):
        import ctypes
        self.ct = ctypes
        self.u = ctypes.windll.user32
        self.w = self.u.GetSystemMetrics(0)
        self.h = self.u.GetSystemMetrics(1)
        self.last = None

    def size(self):
        return self.w, self.h

    def position(self):
        class POINT(self.ct.Structure):
            _fields_ = [("x", self.ct.c_long), ("y", self.ct.c_long)]
        p = POINT()
        self.u.GetCursorPos(self.ct.byref(p))
        return float(p.x), float(p.y)

    def move(self, x, y):
        pos = (int(round(x)), int(round(y)))
        if pos != self.last:
            self.u.SetCursorPos(pos[0], pos[1])
            self.last = pos

    def button(self, name, down):
        flags = {("left", True): 0x0002, ("left", False): 0x0004,
                 ("right", True): 0x0008, ("right", False): 0x0010}
        self.u.mouse_event(flags[(name, down)], 0, 0, 0, 0)

    def wheel(self, delta):
        self.u.mouse_event(0x0800, 0, 0, int(delta), 0)     # positive = page goes up

    def _key(self, vk, up=False, extended=False):
        flags = (0x0002 if up else 0) | (0x0001 if extended else 0)
        self.u.keybd_event(vk, 0, flags, 0)

    def tap(self, vk, extended=False):
        self._key(vk, False, extended)
        self._key(vk, True, extended)

    def volume(self, up):
        self.tap(self.VK_VOL_UP if up else self.VK_VOL_DOWN)

    def arrow(self, right):
        self.tap(self.VK_RIGHT if right else self.VK_LEFT, extended=True)

    def enter(self):
        self.tap(self.VK_RETURN)

    def escape(self):
        self.tap(self.VK_ESCAPE)

    def win_tab(self):
        self._key(self.VK_LWIN)
        time.sleep(0.02)
        self._key(self.VK_TAB)
        self._key(self.VK_TAB, up=True)
        time.sleep(0.02)
        self._key(self.VK_LWIN, up=True)


class DemoActuator:
    """Used when this is not Windows: it only prints what it would do."""

    def __init__(self):
        self.pos = (500.0, 400.0)

    def size(self):
        return 1920, 1080

    def position(self):
        return self.pos

    def move(self, x, y):
        self.pos = (x, y)

    def button(self, name, down):
        print("[demo] mouse", name, "down" if down else "up")

    def wheel(self, delta):
        print("[demo] wheel", delta)

    def volume(self, up):
        print("[demo] volume", "up" if up else "down")

    def arrow(self, right):
        print("[demo] arrow", "right" if right else "left")

    def enter(self):
        print("[demo] Enter")

    def escape(self):
        print("[demo] Esc")

    def win_tab(self):
        print("[demo] Win+Tab")


def make_actuator():
    return WinActuator() if os.name == "nt" else DemoActuator()


# ---------- 5. Measuring one hand ----------
class Hand:
    """Numbers about one hand in one frame: distances, which fingers are straight, positions."""

    def __init__(self, pts, w, h, p_right):
        self.pts = pts                                  # 21 points in pixels
        self.p_right = p_right                          # how sure the network is that it's the RIGHT hand (0..1)
        self.size = max(math.dist(pts[0], pts[9]), 1.0)   # palm length: wrist -> base of the middle finger
        self.d_idx = math.dist(pts[4], pts[8]) / self.size      # thumb tip <-> index tip
        self.d_mid = math.dist(pts[4], pts[12]) / self.size     # thumb tip <-> middle tip
        # how straight each finger is: index, middle, ring, pinky
        self.ext = []
        for tip, knuckle in ((8, 5), (12, 9), (16, 13), (20, 17)):
            self.ext.append(math.dist(pts[0], pts[tip]) / max(math.dist(pts[0], pts[knuckle]), 1.0))
        cx = sum(pts[i][0] for i in (0, 5, 9, 13, 17)) / 5
        cy = sum(pts[i][1] for i in (0, 5, 9, 13, 17)) / 5
        self.palm_n = (cx / w, cy / h)                  # middle of the palm (it doesn't move when you pinch)
        self.tip_n = (pts[8][0] / w, pts[8][1] / h)     # index fingertip
        self.wrist_n = (pts[0][0] / w, pts[0][1] / h)


# ---------- 6. Remembering hands between frames ----------
class Track:
    """Memory about one hand: who it is (Left/Right) and what it is doing now."""

    def __init__(self, tid, now):
        self.id = tid
        self.label = None            # "Left" or "Right" (stable, doesn't flicker)
        self.p = 0.5                 # smoothed chance that it's the right hand
        self.hand = None
        self.visible = False
        self.last_seen = now
        self.wrist_n = (0.5, 0.5)
        # fist / open palm
        self.pose = "none"
        self.pose_since = now
        self.last_fist = -99.0
        self.fist_entered = -99.0
        self.open_entered = -99.0
        self.open_from_fist = False
        # pinch
        self.pinch = None            # "idx" / "mid" while pinching
        self.cand = None
        self.cand_since = 0.0
        self.rel_since = None
        self.btn = None              # mouse button held by this hand
        # left hand modes
        self.mode = None             # "volume" / "scroll"
        self.pf = (0.0, 0.0)
        self.anchor = (0.0, 0.0)
        self.accum = 0.0
        self.mode_text = ""
        # pointing pose and whole-hand pose
        self.point = False
        self.palm = False
        # arrows by hand movement (inside Win+Tab)
        self.sw_anchor = None
        self.sw_pos = (0.0, 0.0)
        # for drawing
        self.action = "idle"
        self.pinch_finger = None
        self.is_driver = False


class Tracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def match(self, hands, now, swap):
        """Connects the hands of this frame with the hands of the previous frame. Returns tracks that are gone."""
        pairs = []
        for ti, t in enumerate(self.tracks):
            for hi, h in enumerate(hands):
                pairs.append((math.dist(t.wrist_n, h.wrist_n), ti, hi))
        pairs.sort()
        used_t, used_h = set(), set()
        for d, ti, hi in pairs:
            if d > MATCH_DIST:
                break
            if ti in used_t or hi in used_h:
                continue
            used_t.add(ti)
            used_h.add(hi)
            self._apply(self.tracks[ti], hands[hi], now, swap, new=False)
        for hi, h in enumerate(hands):
            if hi not in used_h:
                t = Track(self.next_id, now)
                self.next_id += 1
                self.tracks.append(t)
                used_t.add(len(self.tracks) - 1)
                self._apply(t, h, now, swap, new=True)
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.visible = False

        # two hands must not have the same label: the picture is a mirror, so the hand on the left is LEFT
        vis = [t for t in self.tracks if t.visible]
        if len(vis) >= 2 and vis[0].label == vis[1].label:
            a, b = sorted(vis[:2], key=lambda t: t.wrist_n[0])
            a.label, b.label = "Left", "Right"
            a.p, b.p = 0.2, 0.8

        gone = [t for t in self.tracks if not t.visible and now - t.last_seen > GRACE]
        self.tracks = [t for t in self.tracks if t not in gone]
        return gone

    @staticmethod
    def _apply(t, h, now, swap, new):
        obs = 1.0 - h.p_right if swap else h.p_right
        t.p = obs if new else 0.65 * t.p + 0.35 * obs
        if t.label is None:
            t.label = "Right" if t.p > 0.5 else "Left"
        elif t.label == "Right" and t.p < 0.30:
            t.label = "Left"
        elif t.label == "Left" and t.p > 0.70:
            t.label = "Right"
        t.hand = h
        t.visible = True
        t.last_seen = now
        t.wrist_n = h.wrist_n

    def swap_all(self):
        for t in self.tracks:
            t.p = 1.0 - t.p
            t.label = "Left" if t.label == "Right" else "Right"


# ---------- 7. The cursor ----------
class Cursor:
    """
    Three states:
      free - the cursor follows the fingertip
      hold - the cursor stands still (the fingers are closing for a click, or the hand is a fist)
      drag - a mouse button is down: the cursor moves with the hand, but only after a small dead zone
    When we go back to "free" the cursor does not jump: it glides to the fingertip.
    """

    def __init__(self, act):
        self.act = act
        self.W, self.H = act.size()
        self.fx = OneEuro(CURSOR_MIN_CUTOFF, CURSOR_BETA)
        self.fy = OneEuro(CURSOR_MIN_CUTOFF, CURSOR_BETA)
        self.state = "hold"
        self.tx = None
        self.ty = None
        self.corr = (0.0, 0.0)
        self.hist = deque()
        self.anchor = (0.0, 0.0)
        self.ref = (0.0, 0.0)
        self.resync = True
        self.prev_t = None

    def new_driver(self):
        self.resync = True

    def to_screen(self, nx, ny):
        sx = (nx - ACTIVE_X[0]) / (ACTIVE_X[1] - ACTIVE_X[0]) * (self.W - 1)
        sy = (ny - ACTIVE_Y[0]) / (ACTIVE_Y[1] - ACTIVE_Y[0]) * (self.H - 1)
        return clamp(sx, 0, self.W - 1), clamp(sy, 0, self.H - 1)

    def update(self, raw, hold, drag, now):
        if self.tx is None:
            self.tx, self.ty = self.act.position()
        dt = 0.03 if self.prev_t is None else clamp(now - self.prev_t, 0.001, 0.2)
        self.prev_t = now

        fresh = False
        if raw is None:
            want = "hold"
            rx = ry = None
        else:
            if self.resync:                  # a different hand (or the hand came back): start the smoothing again
                self.fx.reset()
                self.fy.reset()
                self.resync = False
                fresh = True
            rx = self.fx(raw[0], now)
            ry = self.fy(raw[1], now)
            want = "drag" if drag else ("hold" if hold else "free")

        prev = self.state

        # free -> hold/drag: go back a little in time, the fingers already moved while closing
        if prev == "free" and want != "free":
            for (t, x, y) in self.hist:
                if t >= now - CURSOR_REWIND:
                    self.tx, self.ty = x, y
                    break
            self.hist.clear()

        # hold/drag -> free (or a new driver hand): remember the gap, then close it slowly
        if want == "free" and (prev != "free" or fresh):
            self.corr = (self.tx - rx, self.ty - ry)

        if want == "drag" and prev != "drag":
            self.anchor = (self.tx, self.ty)
            self.ref = (rx, ry)

        if want == "free":
            k = math.exp(-dt / CURSOR_CATCHUP)
            self.corr = (self.corr[0] * k, self.corr[1] * k)
            self.tx, self.ty = rx + self.corr[0], ry + self.corr[1]
            self.hist.append((now, self.tx, self.ty))
            while self.hist and self.hist[0][0] < now - 0.6:
                self.hist.popleft()
        elif want == "drag":
            vx, vy = rx - self.ref[0], ry - self.ref[1]
            n = math.hypot(vx, vy)
            if n > DRAG_DEADZONE_PX:
                f = (n - DRAG_DEADZONE_PX) / n
                vx, vy = vx * f, vy * f
            else:
                vx = vy = 0.0
            self.tx, self.ty = self.anchor[0] + vx, self.anchor[1] + vy

        self.tx = clamp(self.tx, 2, self.W - 3)
        self.ty = clamp(self.ty, 2, self.H - 3)
        self.state = want
        self.act.move(self.tx, self.ty)


# ---------- 8. The brain: gestures -> actions ----------
class Engine:
    def __init__(self, act):
        self.act = act
        self.cursor = Cursor(act)
        self.tracker = Tracker()
        self.swap = SWAP_HANDS
        self.driver_id = None
        self.tv_open = False
        self.arrow_last = -99.0
        self.last_toggle = -99.0
        self.flash_text = ""
        self.flash_until = 0.0
        self.active = {}             # action -> text, for the screen
        self.t_prev = None

    # ----- helpers -----
    def toggle_swap(self):
        self.swap = not self.swap
        self.tracker.swap_all()
        return self.swap

    def approach(self, t):
        """True if the thumb is coming to the index/middle finger (a pinch is about to happen)."""
        h = t.hand
        return ((h.d_idx < APPROACH and h.ext[0] > PINCH_MIN_EXT) or
                (h.d_mid < APPROACH and h.ext[1] > PINCH_MIN_EXT))

    def stable(self, t, pose, hold, now):
        return t.pose == pose and now - t.pose_since >= hold

    def drop(self, t):
        """A hand left the picture for good: let go of everything it held."""
        if t.btn:
            self.act.button(t.btn, False)
            t.btn = None

    def shutdown(self):
        for t in self.tracker.tracks:
            self.drop(t)

    # ----- one frame -----
    def update(self, hands, now):
        dt = 0.03 if self.t_prev is None else clamp(now - self.t_prev, 0.001, 0.1)
        self.t_prev = now

        for t in self.tracker.match(hands, now, self.swap):
            self.drop(t)
        vis = [t for t in self.tracker.tracks if t.visible]

        for t in vis:
            self.update_pose(t, now)
            t.action = "idle"
            t.pinch_finger = None
            t.is_driver = False

        self.taskview_logic(vis, now, dt)

        for t in vis:
            if t.label == "Right":
                self.right_hand(t, now)
            else:
                self.left_hand(t, now, dt)

        self.cursor_logic(vis, now)
        self.collect_active(vis, now)

    # ----- fists and open palms -----
    def update_pose(self, t, now):
        ext = t.hand.ext
        h = t.hand
        thumb_free = h.d_idx > FIST_THUMB_GAP and h.d_mid > FIST_THUMB_GAP
        if all(e < EXT_DOWN for e in ext) and (thumb_free or max(ext) < FIST_TIGHT):
            pose = "fist"
        elif all(e > EXT_OPEN for e in ext):
            pose = "open"
        else:
            pose = "none"
        if pose != t.pose:
            if pose == "fist":
                t.fist_entered = now
            elif pose == "open":
                t.open_entered = now
                t.open_from_fist = (now - t.last_fist) < FIST_MEMORY
            t.pose = pose
            t.pose_since = now
        if pose == "fist":
            t.last_fist = now

        # "pointing": only the index finger is straight (with a little hysteresis, so it doesn't flicker)
        others = max(ext[1:4])
        if t.point:
            if ext[0] < POINT_STOP_IDX or others > POINT_STOP_OTHER:
                t.point = False
        elif ext[0] > POINT_START_IDX and others < POINT_START_OTHER:
            t.point = True
        # "a whole hand is shown": at least 3 fingers are straight
        t.palm = sum(1 for e in ext if e > PALM_EXT) >= 3

    # ----- Win+Tab -----
    def taskview_logic(self, vis, now, dt):
        paused = now - self.last_toggle < TOGGLE_PAUSE
        if not self.tv_open:
            if not paused and len(vis) >= 2:
                a, b = vis[0], vis[1]
                ok = all(self.stable(t, "open", POSE_HOLD_OPEN, now) and t.open_from_fist
                         and t.open_entered > self.last_toggle for t in (a, b))
                if ok and abs(a.open_entered - b.open_entered) < SYNC_WINDOW:
                    self.open_taskview(vis, now)
            return

        # --- Win+Tab is open ---
        if not paused:
            if len(vis) >= 2:                      # two fists together = close
                a, b = vis[0], vis[1]
                ok = all(self.stable(t, "fist", POSE_HOLD_FIST, now)
                         and t.fist_entered > self.last_toggle for t in (a, b))
                if ok and abs(a.fist_entered - b.fist_entered) < CLOSE_SYNC:
                    self.close_taskview(now)
                    return
            for t in vis:                          # ONE fist (the other hand is not a fist) = Enter
                if (self.stable(t, "fist", ENTER_HOLD, now) and t.fist_entered > self.last_toggle
                        and not any(o.pose == "fist" for o in vis if o is not t)):
                    self.act.enter()
                    self.tv_open = False           # the chosen window opens and Win+Tab closes by itself
                    self.last_toggle = now
                    return
        self.swipes(vis, now, dt)

    def open_taskview(self, vis, now):
        self.act.win_tab()
        self.tv_open = True
        self.last_toggle = now
        for t in vis:
            t.sw_anchor = None
            t.pinch = None
            t.mode = None
            t.cand = None

    def close_taskview(self, now):
        if TASKVIEW_CLOSE == "wintab":
            self.act.win_tab()
        else:
            self.act.escape()
        self.tv_open = False
        self.last_toggle = now

    def swipes(self, vis, now, dt):
        """Inside Win+Tab: ONE open hand moves right = arrow right, moves left = arrow left."""
        palms = [t for t in vis if t.palm and t.pose != "fist"]
        for t in vis:
            if len(palms) != 1 or t is not palms[0]:
                t.sw_anchor = None              # two open hands (or none) = no arrows
        if len(palms) != 1:
            return
        t = palms[0]
        px, py = t.hand.palm_n
        if t.sw_anchor is None:
            t.sw_anchor = (px, py)
            t.sw_pos = (px, py)
            return
        t.sw_pos = (t.sw_pos[0] + (px - t.sw_pos[0]) * PALM_SMOOTH,
                    t.sw_pos[1] + (py - t.sw_pos[1]) * PALM_SMOOTH)
        k = min(1.0, SWIPE_LEAK * dt)           # the start point slowly follows the hand: slow drift does nothing
        t.sw_anchor = (t.sw_anchor[0] + (t.sw_pos[0] - t.sw_anchor[0]) * k,
                       t.sw_anchor[1] + (t.sw_pos[1] - t.sw_anchor[1]) * k)
        dx = t.sw_pos[0] - t.sw_anchor[0]
        dy = t.sw_pos[1] - t.sw_anchor[1]
        if abs(dx) > SWIPE_STEP and abs(dy) < abs(dx) and now - self.arrow_last >= SWIPE_COOLDOWN:
            self.act.arrow(right=(dx > 0))
            self.arrow_last = now
            t.sw_anchor = t.sw_pos             # next arrow needs a new movement
            self.flash_text = "RIGHT >>" if dx > 0 else "<< LEFT"
            self.flash_until = now + 0.5

    # ----- pinch with debounce -----
    def pinch_update(self, t, now):
        """Returns "start" / "end" / None. The pinch is thumb+index ("idx") or thumb+middle ("mid")."""
        h = t.hand
        if t.pinch is None:
            cand = None
            if now - t.last_fist > FIST_LOCKOUT:
                if h.d_idx < PINCH_ON and h.d_idx < h.d_mid * OTHER_RATIO and h.ext[0] > PINCH_MIN_EXT:
                    cand = "idx"
                elif h.d_mid < PINCH_ON and h.d_mid < h.d_idx * OTHER_RATIO and h.ext[1] > PINCH_MIN_EXT:
                    cand = "mid"
            if cand is None or cand != t.cand:
                t.cand = cand
                t.cand_since = now
            elif now - t.cand_since >= PRESS_DELAY:
                t.pinch = cand
                t.cand = None
                t.rel_since = None
                return "start"
        else:
            d = h.d_idx if t.pinch == "idx" else h.d_mid
            if d > PINCH_OFF:
                if t.rel_since is None:
                    t.rel_since = now
                elif now - t.rel_since >= RELEASE_DELAY:
                    t.pinch = None
                    return "end"
            else:
                t.rel_since = None
        return None

    # ----- right hand: mouse buttons -----
    def right_hand(self, t, now):
        ev = self.pinch_update(t, now)
        if ev == "start":
            t.btn = "left" if t.pinch == "idx" else "right"
            self.act.button(t.btn, True)
            if t.btn == "left" and self.tv_open:
                self.tv_open = False      # a click on a window closes Win+Tab by itself
        elif ev == "end" and t.btn:
            self.act.button(t.btn, False)
            t.btn = None
        if t.btn:
            t.action = t.btn
            t.pinch_finger = 8 if t.btn == "left" else 12
        elif self.approach(t):
            t.pinch_finger = 8 if t.hand.d_idx < t.hand.d_mid else 12

    # ----- left hand: volume and scroll -----
    def left_hand(self, t, now, dt):
        if self.tv_open:
            t.mode = None
            t.pinch = None
            return

        ev = self.pinch_update(t, now)
        if ev == "start":
            t.mode = "volume" if t.pinch == "idx" else "scroll"
            t.pf = t.hand.palm_n
            t.anchor = t.pf
            t.accum = 0.0
            t.mode_text = t.mode.upper()
        elif ev == "end":
            t.mode = None

        if t.mode is None:
            if self.approach(t):
                t.pinch_finger = 8 if t.hand.d_idx < t.hand.d_mid else 12
            return

        t.action = t.mode
        t.pinch_finger = 8 if t.mode == "volume" else 12
        px, py = t.hand.palm_n
        t.pf = (t.pf[0] + (px - t.pf[0]) * PALM_SMOOTH, t.pf[1] + (py - t.pf[1]) * PALM_SMOOTH)

        if t.mode == "volume":
            v = (t.pf[0] - t.anchor[0]) - (t.pf[1] - t.anchor[1])      # right and up = plus
            rate = clamp(softdz(v, MODE_DEADZONE) * VOLUME_SPEED, -VOLUME_MAX, VOLUME_MAX)
            t.accum += rate * dt
            n = int(clamp(int(t.accum), -5, 5))
            for _ in range(abs(n)):
                self.act.volume(n > 0)
            t.accum -= n
            t.mode_text = "VOLUME +" if rate > 0.5 else ("VOLUME -" if rate < -0.5 else "VOLUME")
        else:
            v = t.pf[1] - t.anchor[1]                                  # hand lower = page goes up
            clicks = clamp(softdz(v, MODE_DEADZONE) * SCROLL_SPEED, -SCROLL_MAX, SCROLL_MAX)
            t.accum += clicks * 120 * dt
            if abs(t.accum) >= 30:
                d = int(t.accum)
                self.act.wheel(d)
                t.accum -= d
            t.mode_text = "SCROLL UP" if clicks > 0.5 else ("SCROLL DOWN" if clicks < -0.5 else "SCROLL")

    # ----- who moves the cursor -----
    def pick_driver(self, vis):
        """The cursor follows a hand that shows ONE index finger (the other fingers are bent)."""
        cands = [t for t in vis if not t.mode]
        pointing = [t for t in cands if t.point]
        holder = next((t for t in cands if t.btn), None)

        def best(group):
            cur = next((t for t in group if t.id == self.driver_id), None)
            if cur is not None:                 # keep the current hand, so the cursor doesn't jump around
                return cur
            for label in ("Right", "Left"):
                for t in group:
                    if t.label == label:
                        return t
            return group[0]

        if holder is not None:                  # a mouse button is held
            others = [t for t in pointing if t is not holder]
            if others:                          # the OTHER hand points: it moves the cursor (drag and drop)
                return best(others)
            return holder                       # no other hand: the hand that holds the button moves it
        if pointing:
            return best(pointing)
        return None                             # nobody points: the cursor stays where it is

    def cursor_logic(self, vis, now):
        drv = self.pick_driver(vis)
        drv_id = drv.id if drv else None
        if drv_id != self.driver_id:
            self.cursor.new_driver()
            self.driver_id = drv_id
        if drv is None:
            self.cursor.update(None, True, False, now)
            return
        drv.is_driver = True
        raw = self.cursor.to_screen(*drv.hand.tip_n)
        locked = now - drv.last_fist < FIST_LOCKOUT
        hold = self.approach(drv) or locked
        self.cursor.update(raw, hold, drv.btn is not None, now)
        if drv.action == "idle":
            drv.action = "cursor"

    # ----- what to show on the screen -----
    def collect_active(self, vis, now):
        act = {}
        for t in vis:
            if self.tv_open and t.action == "idle":
                t.action = "wintab"            # inside Win+Tab the hands are painted yellow
            if t.btn == "left":
                act["left"] = "LEFT CLICK"
            elif t.btn == "right":
                act["right"] = "RIGHT CLICK"
            if t.mode:
                act[t.mode] = t.mode_text
            if t.is_driver and self.cursor.state == "free":
                act["cursor"] = "CURSOR"
        if self.tv_open:
            act["wintab"] = "WIN+TAB  " + (self.flash_text if now < self.flash_until else "")
        self.active = act

    def primary(self):
        for name in PRIORITY:
            if name in self.active:
                return name
        return "idle"


# ---------- 9. Drawing on the camera picture ----------
def draw_hand(view, t):
    h = t.hand
    color = COLORS[t.action]
    for i in FINGERTIPS:
        x, y = int(h.pts[i][0]), int(h.pts[i][1])
        big = (t.action not in ("idle",)) and (i == 4 or i == t.pinch_finger or (t.action == "cursor" and i == 8))
        s = 15 if big else 9
        cv2.rectangle(view, (x - s, y - s), (x + s, y + s), color, 3 if big else 2)
    if t.pinch_finger is not None:         # a line between the thumb and the finger
        a = (int(h.pts[4][0]), int(h.pts[4][1]))
        b = (int(h.pts[t.pinch_finger][0]), int(h.pts[t.pinch_finger][1]))
        cv2.line(view, a, b, color if t.action != "idle" else COLORS["idle"], 2)
    if t.is_driver:                        # a cross on the fingertip that moves the cursor
        x, y = int(h.pts[8][0]), int(h.pts[8][1])
        cv2.line(view, (x - 22, y), (x + 22, y), COLORS["cursor"], 1)
        cv2.line(view, (x, y - 22), (x, y + 22), COLORS["cursor"], 1)
    wx, wy = int(h.pts[0][0]), int(h.pts[0][1])
    cv2.putText(view, t.label.upper(), (wx - 25, min(wy + 28, view.shape[0] - 70)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_screen(view, eng, show_debug, fps, cam_fps, det_ms, cam_idx):
    hh, ww = view.shape[:2]
    prim = eng.primary()
    border = COLORS[prim]
    cv2.rectangle(view, (0, 0), (ww - 1, hh - 1), border, 10 if prim != "idle" else 2)

    for t in eng.tracker.tracks:
        if t.visible:
            draw_hand(view, t)

    # title and numbers
    cv2.putText(view, "M.E.N.T.A.L. v4.1", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(view, f"FPS {int(fps)}  CAM{cam_idx} {cam_fps}fps  DET {int(det_ms)}ms",
                (14, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 255), 1)

    # active actions: colored labels at the top
    x = 14
    for key, label in NAMES:
        if key in eng.active:
            text = eng.active[key] or label
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(view, (x, 60), (x + tw + 14, 60 + th + 14), COLORS[key], -1)
            cv2.putText(view, text, (x + 7, 60 + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            x += tw + 22

    # debug numbers
    if show_debug:
        y = hh - 70
        for t in eng.tracker.tracks:
            if t.visible:
                h = t.hand
                s = (f"{t.label[0]} {t.pose:5s} dI={h.d_idx:.2f} dM={h.d_mid:.2f} "
                     f"ext={h.ext[0]:.2f} {h.ext[1]:.2f} {h.ext[2]:.2f} {h.ext[3]:.2f}")
                cv2.putText(view, s, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                y -= 18

    # legend at the bottom: all colors
    x = 14
    y = hh - 26
    for key, label in NAMES:
        on = key in eng.active
        c = COLORS[key]
        if on:
            cv2.rectangle(view, (x - 2, y - 14), (x + 14, y + 2), c, -1)
            cv2.rectangle(view, (x - 2, y - 14), (x + 14, y + 2), (255, 255, 255), 1)
        else:
            cv2.rectangle(view, (x, y - 12), (x + 12, y), c, -1)
        cv2.putText(view, label, (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255) if on else (150, 150, 150), 1)
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        x += tw + 34
    cv2.putText(view, "q exit | c camera | h swap hands | d debug", (14, hh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


# ---------- 10. Shared variables (all threads can see them) ----------
running = True
lock = threading.Lock()
latest_frame = None
frame_id = 0
display_frame = None
cam_fps = 0
cam_index = CAMERA_NUMBER
switch_camera = False
want_swap = False
show_debug = False
engine = None


def open_camera(index):
    """Opens a camera. Returns None if it doesn't work."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # compressed format: gives more FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                              # don't pile up old frames
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    print("Camera", index, "gave:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x",
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "@", cap.get(cv2.CAP_PROP_FPS), "FPS")
    return cap


def camera_loop(start_index):
    """Camera thread: only grabs frames. The "c" key switches to the next camera here."""
    global latest_frame, frame_id, running, cam_fps, cam_index, switch_camera
    cap = open_camera(start_index)
    if cap is None:
        print(f"Could not open camera {start_index}. Try --camera 1 (or 2) or change CAMERA_NUMBER in the settings.")
        running = False
        return
    cam_index = start_index
    count = 0
    fails = 0
    t0 = time.time()
    while running:
        if switch_camera:
            switch_camera = False
            for step in range(1, MAX_CAMERAS):
                idx = (cam_index + step) % MAX_CAMERAS
                new = open_camera(idx)
                if new is not None:
                    cap.release()
                    cap = new
                    cam_index = idx
                    break
            else:
                print("No other camera found")
        ok, frame = cap.read()
        if not ok:
            fails += 1
            if fails > 30:
                print("The camera stopped giving frames")
                running = False
                break
            time.sleep(0.01)
            continue
        fails = 0
        frame = cv2.flip(frame, 1)          # mirror it, so it looks like a mirror
        with lock:
            latest_frame = frame
            frame_id += 1
        count += 1
        if time.time() - t0 >= 1:
            cam_fps = count
            count = 0
            t0 = time.time()
    cap.release()


def detect_loop():
    """Detection thread: finds the hands, runs the gestures, moves the mouse."""
    global display_frame, want_swap
    if not os.path.exists(MODEL_FILE):
        print("Downloading the hand model (one time only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print("Done!")

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_FILE),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    last_id = -1
    last_ts = 0
    fps = 0.0
    det_ms = 0.0
    prev_time = time.perf_counter()

    while running:
        with lock:
            frame = latest_frame
            fid = frame_id
        if frame is None or fid == last_id:
            time.sleep(0.002)
            continue
        last_id = fid

        h, w, _ = frame.shape
        small = cv2.resize(frame, (DETECT_WIDTH, int(h * DETECT_WIDTH / w)))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.time() * 1000), last_ts + 1)
        last_ts = ts
        t0 = time.perf_counter()
        result = detector.detect_for_video(image, ts)
        det_ms = det_ms * 0.9 + (time.perf_counter() - t0) * 1000 * 0.1

        hands = []
        for i, lms in enumerate(result.hand_landmarks):
            cat = result.handedness[i][0]
            p_right = cat.score if cat.category_name == "Right" else 1.0 - cat.score
            pts = [(p.x * w, p.y * h) for p in lms]
            hands.append(Hand(pts, w, h, p_right))

        if want_swap:
            want_swap = False
            print("Hands swapped. SWAP_HANDS =", engine.toggle_swap())

        now = time.perf_counter()
        engine.update(hands, now)

        view = frame.copy()
        dt = max(now - prev_time, 0.001)
        fps = fps * 0.9 + (1 / dt) * 0.1
        prev_time = now
        draw_screen(view, engine, show_debug, fps, cam_fps, det_ms, cam_index)
        with lock:
            display_frame = view


def detect_thread():
    """Wrapper: if the detection hits an error, we will see it instead of silently freezing."""
    global running
    try:
        detect_loop()
    except Exception:
        traceback.print_exc()
    running = False
    if engine is not None:
        engine.shutdown()      # whatever happens, the mouse buttons must not stay pressed


# ---------- 11. Start ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M.E.N.T.A.L. - control the computer with your hands")
    parser.add_argument("--camera", type=int, default=CAMERA_NUMBER, help="camera number (0, 1, 2...)")
    parser.add_argument("--no-window", action="store_true", help="run without the camera window")
    args = parser.parse_args()
    if args.no_window:
        SHOW_WINDOW = False

    engine = Engine(make_actuator())
    WIN = "M.E.N.T.A.L."

    threading.Thread(target=camera_loop, args=(args.camera,), daemon=True).start()
    threading.Thread(target=detect_thread, daemon=True).start()

    window_ready = False
    try:
        while running:
            if SHOW_WINDOW:
                with lock:
                    view = display_frame
                if view is not None:
                    if not window_ready:
                        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
                        if PREVIEW_ON_TOP:
                            try:
                                cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)
                            except Exception:
                                pass
                        window_ready = True
                    cv2.imshow(WIN, view)
                key = cv2.waitKey(15) & 0xFF
                if key in (ord("q"), 27):
                    running = False
                elif key == ord("c"):
                    switch_camera = True
                elif key == ord("h"):
                    want_swap = True
                elif key == ord("d"):
                    show_debug = not show_debug
                if window_ready and cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    running = False          # the window was closed with the X button
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        running = False             # Ctrl+C in the terminal stops it too

    running = False
    time.sleep(0.3)
    engine.shutdown()
    cv2.destroyAllWindows()
