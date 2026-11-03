# M.E.N.T.A.L.

Control your computer mouse with just your hands, using a regular webcam.

The program tracks your hands in real time, draws squares on your fingertips and turns gestures into mouse actions.

## Features

- Real-time hand tracking (21 landmarks per hand, MediaPipe)
- Squares follow your fingertips on the camera preview
- Cursor control: close all fingers together and move your hand
- Left click: open hand + thumb and index finger pinch
- Right click: thumb and middle finger pinch
- Hold & drag: the button stays pressed while you keep the gesture, so you can select text and drag windows
- Works with both hands
- The cursor stays where you released it
- Works even when the app window is minimized
- Camera runs in a separate thread for lower lag

## Tech stack

- Python
- MediaPipe
- OpenCV

## How to run

1. Install Python 3
2. Install dependencies:

```
pip install opencv-python mediapipe pyautogui
```

3. Go into the project folder and run the program:

```
cd mental
python M.E.N.T.A.L..py
```

Note: the file `hand_landmarker.task` (the MediaPipe hand model) must stay in the same folder as the program.

## Plans

- Train a custom small neural network for custom gestures
