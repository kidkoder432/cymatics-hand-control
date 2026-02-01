import cv2
import mediapipe as mp
import numpy as np
import mido
from time import perf_counter_ns

# --- MIDI SETUP ---
# Ensure you have a port named 'PythonOut' in loopMIDI
try:
    print(mido.get_output_names())
    outport = mido.open_output("PythonOut 1")
    print("Connected to loopMIDI: PythonOut")
except:
    print("loopMIDI port 'PythonOut' not found. Using default output.")
    outport = mido.open_output()

MAX_VOICES = 2
last_midi_note = [None] * MAX_VOICES


def send_pitch_bend(voice_idx, raw_midi_val):
    """
    Calculates the pitch bend needed to reach the fractional MIDI note.
    Assumes Decent Sampler pitch bend range is set to +/- 12 semitones.
    """
    nearest = round(raw_midi_val)
    bend_amount = raw_midi_val - nearest
    # MIDI Pitch Bend is 14-bit (0 to 16383). 8192 is absolute center.
    # Formula: center + (fractional_offset / semitones_range) * range_max
    bend_value = int(-8192 + (bend_amount * 8192 / 12))
    outport.send(
        mido.Message(
            "pitchwheel", channel=voice_idx, pitch=max(0, min(bend_value, 16383))
        )
    )


# --- MEDIAPIPE SETUP ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=MAX_VOICES,
)
hand_landmarker = HandLandmarker.create_from_options(options)


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

    # Track which voices are active this frame
    active_voices = [False] * MAX_VOICES

    if result.hand_landmarks:
        for i, hand in enumerate(result.hand_landmarks):
            if result.handedness[i][0].category_name == "Right":
                continue
            if i >= MAX_VOICES:
                break
            active_voices[i] = True

            # 1. VOLUME/EXPRESSION (Hand Openness)
            # Map distance between Wrist (0) and Middle Tip (12) to MIDI CC 7
            wrist, middle_tip = hand[0], hand[12]
            dist = np.sqrt(
                (middle_tip.x - wrist.x) ** 2 + (middle_tip.y - wrist.y) ** 2
            )
            # Send CC 7 (Volume) to Decent Sampler
            vol = int(scale(dist, 0.1, 0.4, 0, 127))
            outport.send(
                mido.Message(
                    "control_change", channel=i, control=7, value=max(0, min(vol, 127))
                )
            )

            # 2. PITCH LOGIC (Vertical/Y position)
            avg_y = sum(lm.y for lm in hand) / len(hand)
            clamped_y = 1 - max(0, min(avg_y, 1))

            # Map Y to a MIDI Range (e.g., MIDI 48 to 84)
            raw_midi_note = scale(clamped_y, 0, 1, 48, 84)
            snapped_midi = round(raw_midi_note)

            # 3. SNAP-TO-NOTE vs GLIDE
            # Threshold in semitones (0.15 = 15% of the way to the next note)
            diff = abs(raw_midi_note - snapped_midi)
            threshold = 0.2

            # Trigger New Note if it changed
            if last_midi_note[i] != snapped_midi:
                if last_midi_note[i] is not None:
                    outport.send(
                        mido.Message("note_off", note=last_midi_note[i], channel=i)
                    )
                outport.send(
                    mido.Message("note_on", note=snapped_midi, velocity=90, channel=i)
                )
                last_midi_note[i] = snapped_midi

            # Apply Glide or Reset Pitch Wheel
            if diff > threshold:
                send_pitch_bend(i, raw_midi_note)
                label = "Glide"
            else:
                outport.send(mido.Message("pitchwheel", channel=i, pitch=0))
                label = f"Note: {snapped_midi}"

            # UI Feedback
            cv2.putText(
                frame,
                f"H{i} {label}",
                (50, 50 + (i * 40)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

    # 4. CLEANUP: Turn off notes for hands that disappeared
    for i in range(MAX_VOICES):
        if not active_voices[i] and last_midi_note[i] is not None:
            outport.send(mido.Message("note_off", note=last_midi_note[i], channel=i))
            last_midi_note[i] = None

    cv2.imshow("MIDI Hand Controller", cv2.flip(frame, 1))
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
