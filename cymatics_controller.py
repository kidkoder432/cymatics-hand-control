import cv2
from matplotlib.pylab import f
import mediapipe as mp
import numpy as np
import mido
from time import perf_counter_ns

import serial

ser = serial.Serial('COM15', 115200)

# --- MIDI SETUP ---
# Ensure you have a port named 'PythonOut' in loopMIDI
try:
    print(mido.get_output_names())
    outport = mido.open_output("PythonOut 1")
    print("Connected to loopMIDI: PythonOut")
except:
    print("loopMIDI port 'PythonOut' not found. Using default output.")
    outport = mido.open_output()


# --- MEDIAPIPE SETUP ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
)
hand_landmarker = HandLandmarker.create_from_options(options)

def get_openness(hand):
    wrist, middle_tip = hand[0], hand[12]
    dist = np.sqrt((middle_tip.x - wrist.x) ** 2 + (middle_tip.y - wrist.y) ** 2)
    return dist

# 0 is bottom!
def get_coords(hand):
    avg_x = sum(lm.x for lm in hand) / len(hand)
    avg_y = sum(lm.y for lm in hand) / len(hand)
    return (avg_x, 1 - avg_y)

def scale(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp = int(perf_counter_ns() / 1_000_000)
    result = hand_landmarker.detect_for_video(mp_image, timestamp)

    frame = cv2.flip(frame, 1)


    if len(result.hand_landmarks) == 2:
        lx, ly = get_coords(result.hand_landmarks[0])
        rx, ry = get_coords(result.hand_landmarks[1])
        lop = get_openness(result.hand_landmarks[0])
        rop = get_openness(result.hand_landmarks[1])

        print(lx, ly, rx, ry, lop, rop, sep=", ")
        ser.write(f"{lx:.2f},{ly:.2f},{lop:.2f},{rx:.2f},{ry:.2f},{rop:.2f}\n".encode())
        cv2.putText(frame, f"Left: ({lx:.2f}, {ly:.2f}, Openness: {lop:.2f}), Right: ({rx:.2f}, {ry:.2f}, Openness: {rop:.2f})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "WARNING: Please use both hands for the controller to work properly.", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("CYMATIC Hand Controller", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break
    if cv2.getWindowProperty('CYMATIC Hand Controller', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
