# Mental Control v4

Control your computer using only your hands and a webcam!

## Features

- Mouse control with your index finger.
- Left and right mouse clicks.
- Drag and drop files and windows using both hands.
- System volume control.
- Page scrolling.
- Windows Task View (Win + Tab) control.
- Real-time gesture visualization with different colors.
- Both hands can work simultaneously.

## Hand Controls

### Right Hand
- **Left Click:** Touch your thumb and index finger.
- **Right Click:** Touch your thumb and middle finger.
- **Drag:** Hold your thumb and index finger together while moving the cursor.

### Left Hand
- **Volume Control:** Touch your thumb and index finger.
  - Move your hand up or right to increase volume.
  - Move your hand down or left to decrease volume.
- **Scrolling:** Touch your thumb and middle finger.
  - Move your hand down to scroll up.
  - Move your hand up to scroll down.

### Mouse Cursor
- Show only your index finger, keeping the other fingers folded.
- The cursor follows your index fingertip.
- You can use either hand.

### Drag & Drop
1. Point at an object with your left index finger.
2. Touch your right thumb and index finger to hold the mouse button.
3. Move your left hand to drag the object.
4. Release your right fingers to drop it.

You can also drag objects using one hand.

### Windows Task View (Win + Tab)

- **Open:** Open both hands from fists into open palms simultaneously.
- **Close:** Make fists with both hands.
- **Move Right:** Quickly move one open hand to the right.
- **Move Left:** Quickly move one open hand to the left.
- **Enter:** Make a fist with one hand for approximately 0.5 seconds.

Only one hand should control navigation at a time.

## Keyboard Shortcuts

- **H:** Switch or reverse hand recognition if the hands are detected incorrectly.
- **D:** Show finger distance values for gesture calibration.
- **Q:** Quit the application.

## Installation

1. Download and install [Python](https://www.python.org/downloads/).
2. During installation, enable **Add Python to PATH**.
3. Open the project folder.
4. Type `cmd` in the folder's address bar and press Enter.
5. Install the required libraries:

```bash
py -m pip install opencv-python mediapipe pyautogui
```

6. Wait for the installation to finish.

## How to Start

1. Open the project folder.
2. Double-click `START_MENTAL.bat`.
3. Wait for the camera to start.
4. Select your camera if needed.
5. Start controlling your computer!

## Tips

- Use good lighting.
- Keep your hands visible to the camera.
- Avoid very fast movements.
- Watch the colored indicators to see which gesture is detected.
