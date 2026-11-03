# M.E.N.T.A.L. - version 6
# Camera, detection and window run in separate threads.
# Both hands work. Gestures (for either hand):
#   pinch (all fingers together, like "lips") = "holding the mouse": the cursor moves with your hand
#   open your hand = let go of the mouse, the cursor stays where it was
#   thumb + index finger = left button held while you hold the pose (release = button up, so you can drag)
#   thumb + middle finger = right button held while you hold the pose

import os                       # working with files on the computer
import math                     # math (distances)
import time                     # time
import threading                # threads: doing several things at once
import traceback                # to print errors nicely
import urllib.request           # downloading files from the internet
import cv2                      # OpenCV: camera and drawing
import mediapipe as mp          # the neural network that finds the hand
import pyautogui                # mouse control
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------- 0. Stop Windows from slowing the program down when the window is minimized ----------
# (This is a Windows-specific helper block, you can skip it.)
# Windows 11 likes to "save power" and slow down minimized programs.
# Because of that, the pinch stopped working when minimized. Here we forbid it.
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
MODEL_FILE = "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

if not os.path.exists(MODEL_FILE):
    print("Downloading the hand model (one time only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    print("Done!")

# ---------- 2. Settings (you can tweak everything here) ----------
CAMERA_NUMBER = 0        # camera number (0 = first, 1 = second, e.g. Camo)
CAM_WIDTH = 640          # camera frame width
CAM_HEIGHT = 480         # camera frame height
CAM_FPS = 60             # ask the camera for 60 frames per second (if it can)
DETECT_WIDTH = 320       # the neural network looks at a shrunk picture - it's faster
SHOW_WINDOW = True       # show the camera window (False = runs without any window)
NUM_HANDS = 2            # how many hands we track

FINGERTIPS = [4, 8, 12, 16, 20]  # fingertips: thumb, index, middle, ring, pinky
SQUARE_SIZE = 14         # half of the square's side (pixels)
SMOOTH = 0.5             # square smoothing: 0 = very smooth, 1 = no smoothing

GRAB_ON = 0.45           # all fingers closer than this to the thumb = "holding the mouse"
GRAB_OFF = 0.65          # farther than this = let go of the mouse
CLICK_CLOSE = 0.30       # finger closer than this to the thumb = press
CLICK_OPEN = 0.45        # farther than this = finger released
OTHER_FAR = 0.50         # the other fingers must be farther than this, otherwise the button won't press
PRESS_DELAY = 0.10       # how many seconds you must hold the fingers for the button to press (protects from accidents)
CLICK_PAUSE = 0.4        # pause after "let go of the mouse" so it doesn't click by accident

GAIN = 1.8               # cursor speed: bigger = the cursor runs faster after your hand
HAND_SMOOTH = 0.5        # hand smoothing: smaller = smoother, but with a delay
DRAG_DEADZONE = 0.015    # after a button press the cursor stands still until the hand moves this much (so a click doesn't turn into a drag)

pyautogui.PAUSE = 0      # by default pyautogui waits 0.1 sec after every action - that's lag, so we turn it off
SCREEN_W, SCREEN_H = pyautogui.size()   # size of your screen

# ---------- 3. Shared variables (all threads can see them) ----------
running = True           # while True the program keeps working
lock = threading.Lock()  # a "lock": so two threads don't grab the same variable at the same time
latest_frame = None      # the newest frame from the camera
frame_id = 0             # frame number (so we don't process the same frame twice)
display_frame = None     # the finished picture with squares for the window
cam_fps = 0              # how many frames per second the camera really gives

# which hands are holding the mouse buttons right now (so the two hands don't interfere)
holders = {"left": set(), "right": set()}


def dist(a, b):
    """Distance between two points of the hand."""
    return math.hypot(a.x - b.x, a.y - b.y)


def clamp(value, low, high):
    """Keeps the number inside the limits low..high."""
    return max(low, min(high, value))


def press(button, who):
    """Press a mouse button down (unless another hand is already holding it)."""
    if not holders[button]:
        pyautogui.mouseDown(button=button)
    holders[button].add(who)


def release(button, who):
    """Release a mouse button (if nobody else is holding it anymore)."""
    holders[button].discard(who)
    if not holders[button]:
        pyautogui.mouseUp(button=button)


def release_all():
    """Release absolutely everything (just in case, on exit)."""
    for button in holders:
        if holders[button]:
            holders[button].clear()
            pyautogui.mouseUp(button=button)


def new_hand_state():
    """Memory about one hand: what it is doing right now."""
    return {
        "grab": False,        # pinch (holding the mouse)
        "left": False,        # this hand is holding the left button
        "right": False,       # this hand is holding the right button
        "left_since": None,   # since when the fingers have been together for the left button
        "right_since": None,  # the same for the right button
        "cooldown": 0.0,      # until what time clicking is not allowed
        "fx": None,           # smoothed hand position, x
        "fy": None,           # smoothed hand position, y
        "press_x": 0.0,       # where the hand was at the moment of the press
        "press_y": 0.0,
        "free_move": True,    # whether the cursor may move (False = dead zone after a press)
        "squares": {},        # smoothed positions of the squares
    }


def stop_buttons(key, st):
    """If this hand was holding buttons - release them."""
    if st["left"]:
        release("left", key)
        st["left"] = False
    if st["right"]:
        release("right", key)
        st["right"] = False
    st["left_since"] = None
    st["right_since"] = None


def process_hand(key, st, hand, now):
    """Look at what the hand is doing. Returns how far it moved since the last frame."""
    size = max(dist(hand[0], hand[9]), 0.001)   # hand size (wrist -> base of the middle finger)

    # distances from the thumb tip to the other fingertips (relative to hand size)
    d_idx = dist(hand[4], hand[8]) / size
    d_mid = dist(hand[4], hand[12]) / size
    d_ring = dist(hand[4], hand[16]) / size
    d_pinky = dist(hand[4], hand[20]) / size
    farthest = max(d_idx, d_mid, d_ring, d_pinky)

    # hand position (point 9 is the middle of the palm) and how much it moved
    if st["fx"] is None:
        st["fx"], st["fy"] = hand[9].x, hand[9].y
        dx = dy = 0.0
    else:
        nx = st["fx"] + (hand[9].x - st["fx"]) * HAND_SMOOTH
        ny = st["fy"] + (hand[9].y - st["fy"]) * HAND_SMOOTH
        dx, dy = nx - st["fx"], ny - st["fy"]
        st["fx"], st["fy"] = nx, ny

    # --- pinch (it has priority over the buttons) ---
    if not st["grab"] and farthest < GRAB_ON:
        stop_buttons(key, st)
        st["grab"] = True
        st["free_move"] = True
    elif st["grab"] and farthest > GRAB_OFF:
        st["grab"] = False
        st["cooldown"] = now + CLICK_PAUSE

    # --- mouse buttons (only if not pinching) ---
    if not st["grab"]:
        # Left button: thumb + index
        left_pose = d_idx < CLICK_CLOSE and d_mid > OTHER_FAR
        if st["left"]:
            if d_idx > CLICK_OPEN:               # fingers apart - release
                release("left", key)
                st["left"] = False
        elif left_pose and not st["right"] and now > st["cooldown"]:
            if st["left_since"] is None:
                st["left_since"] = now           # start counting how long we hold
            elif now - st["left_since"] >= PRESS_DELAY:
                press("left", key)
                st["left"] = True
                st["left_since"] = None
                st["press_x"], st["press_y"] = st["fx"], st["fy"]
                st["free_move"] = False
        else:
            st["left_since"] = None

        # Right button: thumb + middle
        right_pose = d_mid < CLICK_CLOSE and d_idx > OTHER_FAR
        if st["right"]:
            if d_mid > CLICK_OPEN:
                release("right", key)
                st["right"] = False
        elif right_pose and not st["left"] and now > st["cooldown"]:
            if st["right_since"] is None:
                st["right_since"] = now
            elif now - st["right_since"] >= PRESS_DELAY:
                press("right", key)
                st["right"] = True
                st["right_since"] = None
                st["press_x"], st["press_y"] = st["fx"], st["fy"]
                st["free_move"] = False
        else:
            st["right_since"] = None

    return dx, dy


# ---------- 4. Camera thread: only grabs frames, nothing else ----------
def camera_loop(cap):
    global latest_frame, frame_id, running, cam_fps
    count = 0
    t0 = time.time()
    while running:
        ok, frame = cap.read()
        if not ok:
            print("The camera stopped giving frames")
            running = False
            break
        frame = cv2.flip(frame, 1)      # mirror it, so it looks like a mirror
        with lock:
            latest_frame = frame        # put the newest frame (the old one is thrown away)
            frame_id += 1
        count += 1
        if time.time() - t0 >= 1:       # once a second, count how many frames the camera gave
            cam_fps = count
            count = 0
            t0 = time.time()
    cap.release()


# ---------- 5. Detection thread: finds the hands and controls the mouse ----------
def detect_loop():
    global display_frame

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_FILE),
        running_mode=vision.RunningMode.VIDEO,   # video mode: the network "follows" the hand
        num_hands=NUM_HANDS,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    states = {}              # memory about each hand ("Left" / "Right")
    driver = None            # which hand is moving the cursor right now
    cursor_x = 0.0
    cursor_y = 0.0
    last_id = -1
    last_ts = 0
    fps = 0.0
    det_ms = 0.0
    prev_time = time.time()

    while running:
        # take the newest frame
        with lock:
            frame = latest_frame
            fid = frame_id
        if frame is None or fid == last_id:
            time.sleep(0.002)            # no new frame - wait a little
            continue
        last_id = fid

        h, w, _ = frame.shape
        small = cv2.resize(frame, (DETECT_WIDTH, int(h * DETECT_WIDTH / w)))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts = max(int(time.time() * 1000), last_ts + 1)   # frame time in ms, it must always grow
        last_ts = ts
        t0 = time.time()
        result = detector.detect_for_video(image, ts)
        det_ms = det_ms * 0.9 + (time.time() - t0) * 1000 * 0.1   # how many ms the network thought

        view = frame.copy()      # draw on a copy, so we don't spoil the original
        now = time.time()

        # --- which hands are in the frame (Left / Right) ---
        present = []
        seen = set()
        for i, hand in enumerate(result.hand_landmarks):
            label = result.handedness[i][0].category_name
            if label in seen:                    # if the network named both hands the same
                label = "Left" if label == "Right" else "Right"
            seen.add(label)
            present.append((label, hand))

        # hands that left the frame: release their buttons and forget them
        keys_now = [k for k, _ in present]
        for key in list(states):
            if key not in keys_now:
                stop_buttons(key, states[key])
                del states[key]

        # --- work out the gestures of each hand ---
        moves = {}
        for key, hand in present:
            st = states.setdefault(key, new_hand_state())
            moves[key] = process_hand(key, st, hand, now)

        # --- choose the hand that moves the cursor ---
        active = [k for k in keys_now
                  if states[k]["grab"] or states[k]["left"] or states[k]["right"]]
        if driver not in active:
            driver = active[0] if active else None
            if driver is not None:
                cursor_x, cursor_y = pyautogui.position()   # start from where the cursor is right now

        if driver is not None:
            st = states[driver]
            dx, dy = moves[driver]
            # after a button press the cursor stands still until the hand moves noticeably
            if not st["free_move"]:
                if math.hypot(st["fx"] - st["press_x"], st["fy"] - st["press_y"]) > DRAG_DEADZONE:
                    st["free_move"] = True
            if st["free_move"]:
                # the cursor moves as much as the hand moved (like you're pushing a real mouse)
                cursor_x = clamp(cursor_x + dx * SCREEN_W * GAIN, 5, SCREEN_W - 6)
                cursor_y = clamp(cursor_y + dy * SCREEN_H * GAIN, 5, SCREEN_H - 6)
                pyautogui.moveTo(int(cursor_x), int(cursor_y))

        # --- draw the squares ---
        for key, hand in present:
            st = states[key]
            if st["grab"]:
                color = (0, 255, 255)        # yellow = holding the mouse
            elif st["left"] or st["right"]:
                color = (0, 0, 255)          # red = a button is held
            else:
                color = (0, 255, 0)          # green = normal mode
            for i in FINGERTIPS:
                x = hand[i].x * w
                y = hand[i].y * h
                if i in st["squares"]:
                    old_x, old_y = st["squares"][i]
                    x = old_x + (x - old_x) * SMOOTH
                    y = old_y + (y - old_y) * SMOOTH
                st["squares"][i] = (x, y)
                x, y = int(x), int(y)
                cv2.rectangle(view, (x - SQUARE_SIZE, y - SQUARE_SIZE),
                              (x + SQUARE_SIZE, y + SQUARE_SIZE), color, 2)

        # --- text labels in the window ---
        dt = max(now - prev_time, 0.001)
        fps = fps * 0.9 + (1 / dt) * 0.1     # smoothed FPS, so the number doesn't jump around
        prev_time = now

        parts = []
        if any(states[k]["grab"] for k in keys_now):
            parts.append("HOLD")
        if any(states[k]["left"] for k in keys_now):
            parts.append("LEFT DOWN")
        if any(states[k]["right"] for k in keys_now):
            parts.append("RIGHT DOWN")
        status = " + ".join(parts) if parts else "idle"

        cv2.putText(view, "M.E.N.T.A.L.  (q = exit)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(view, f"FPS: {int(fps)}  CAM: {cam_fps}  DET: {int(det_ms)}ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(view, status, (10, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        with lock:
            display_frame = view     # hand the picture over to the window


def detect_thread():
    """Wrapper: if the detection hits an error, we will see it instead of silently freezing."""
    global running
    try:
        detect_loop()
    except pyautogui.FailSafeException:
        print("Emergency stop: the mouse went to a screen corner")
    except Exception:
        traceback.print_exc()
    running = False
    release_all()            # whatever happens, the mouse buttons must not stay pressed


# ---------- 6. Start ----------
if __name__ == "__main__":
    cap = cv2.VideoCapture(CAMERA_NUMBER, cv2.CAP_DSHOW)   # DSHOW - a fast way to open the camera on Windows
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_NUMBER)
    if not cap.isOpened():
        print("Could not open the camera. Change CAMERA_NUMBER in the settings (0, 1 or 2)")
        raise SystemExit

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # compressed format: gives more FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                              # don't pile up old frames
    print("Camera gave:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x",
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "@", cap.get(cv2.CAP_PROP_FPS), "FPS")

    threading.Thread(target=camera_loop, args=(cap,), daemon=True).start()
    threading.Thread(target=detect_thread, daemon=True).start()

    # the main thread is busy only with the window
    try:
        while running:
            if SHOW_WINDOW:
                with lock:
                    view = display_frame
                if view is not None:
                    cv2.imshow("M.E.N.T.A.L.", view)
                if cv2.waitKey(15) & 0xFF == ord("q"):
                    running = False
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        running = False             # Ctrl+C in the terminal stops it too

    running = False
    time.sleep(0.3)
    release_all()
    cv2.destroyAllWindows()