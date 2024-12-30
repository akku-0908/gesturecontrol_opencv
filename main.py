import cv2
import numpy as np
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL, GUID

# Define thresholds for "Near" and "Far" based on bounding box size
NEAR_THRESHOLD = 30000  # Adjust based on your camera resolution
FAR_THRESHOLD = 10000  # Adjust based on your camera resolution

# Manually define IID_IAudioEndpointVolume
IID_IAudioEndpointVolume = GUID("{5CDF2C82-841E-4546-9722-0CF74078229A}")

def calculate_bounding_box_size(hand_landmarks, frame_shape):
    """Calculate the bounding box size of the hand landmarks."""
    h, w, _ = frame_shape
    x_min = w
    x_max = 0
    y_min = h
    y_max = 0

    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)

    # Calculate the area of the bounding box
    box_width = x_max - x_min
    box_height = y_max - y_min
    box_area = box_width * box_height
    return box_area, (x_min, y_min, x_max, y_max)

def determine_distance(box_area):
    """Determine the hand distance based on bounding box area."""
    if box_area >= NEAR_THRESHOLD:
        return "Near"
    elif box_area <= FAR_THRESHOLD:
        return "Far"
    else:
        return "Neutral"

def get_distance(landmark1, landmark2):
    """Calculate the distance between two landmarks."""
    x1, y1 = landmark1
    x2, y2 = landmark2
    return hypot(x2 - x1, y2 - y1)

def main():
    # Set up MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.75, min_tracking_confidence=0.75)

    # Set up audio controls
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IID_IAudioEndpointVolume, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volRange = volume.GetVolumeRange()
    minVol, maxVol, _ = volRange

    # Open the webcam
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Flip and process the frame
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            left_hand = None
            right_hand = None

            if results.multi_hand_landmarks and results.multi_handedness:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Determine if the hand is left or right
                    hand_label = results.multi_handedness[idx].classification[0].label
                    if hand_label == "Left":
                        left_hand = hand_landmarks
                    elif hand_label == "Right":
                        right_hand = hand_landmarks

            # Process left hand for volume control
            if left_hand:
                # Draw the landmarks
                mp_draw.draw_landmarks(frame, left_hand, mp_hands.HAND_CONNECTIONS)

                # Calculate the bounding box size and get the box coordinates
                box_area, (x_min, y_min, x_max, y_max) = calculate_bounding_box_size(left_hand, frame.shape)

                # Determine the distance status
                distance_status = determine_distance(box_area)

                # Display hand status and bounding box area
                cv2.putText(frame, f"Left Hand: {distance_status}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Box Area: {box_area}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if distance_status == "Near":
                    thumb_tip = left_hand.landmark[4]  # Thumb tip
                    index_tip = left_hand.landmark[8]  # Index finger tip

                    h, w, _ = frame.shape
                    thumb_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                    index_coords = (int(index_tip.x * w), int(index_tip.y * h))

                    # Calculate distance between thumb and index finger
                    distance = get_distance(thumb_coords, index_coords)

                    # Adjust volume
                    volume_level = np.interp(distance, [20, 200], [minVol, maxVol])
                    volume.SetMasterVolumeLevel(volume_level, None)
                    volume_percent = int(np.interp(volume_level, [minVol, maxVol], [0, 100]))
                    cv2.putText(frame, f"Volume: {volume_percent}%", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Process right hand for brightness control
            if right_hand:
                # Draw the landmarks
                mp_draw.draw_landmarks(frame, right_hand, mp_hands.HAND_CONNECTIONS)

                # Calculate the bounding box size and get the box coordinates
                box_area, (x_min, y_min, x_max, y_max) = calculate_bounding_box_size(right_hand, frame.shape)

                # Determine the distance status
                distance_status = determine_distance(box_area)

                # Display hand status and bounding box area
                cv2.putText(frame, f"Right Hand: {distance_status}", (10, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Box Area: {box_area}", (10, 290),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if distance_status == "Near":
                    thumb_tip = right_hand.landmark[4]  # Thumb tip
                    index_tip = right_hand.landmark[8]  # Index finger tip

                    h, w, _ = frame.shape
                    thumb_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                    index_coords = (int(index_tip.x * w), int(index_tip.y * h))

                    # Calculate distance between thumb and index finger
                    distance = get_distance(thumb_coords, index_coords)

                    # Adjust brightness
                    brightness_level = int(np.interp(distance, [20, 200], [0, 100]))
                    sbc.set_brightness(brightness_level)
                    cv2.putText(frame, f"Brightness: {brightness_level}%", (10, 350),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Show the frame
            cv2.imshow("Hand Control", frame)

            # Exit on pressing 'ESC'
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()