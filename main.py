import cv2
import mediapipe as mp
import numpy as np
import time
import math

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the Hands object
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Access the webcam
cap = cv2.VideoCapture(0)

# Initialize time variables for FPS calculation
prev_time = 0
curr_time = 0

# Create a list to store drawing points
drawing_points = []

# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Process the frame for segmentation
    segmentation_results = selfie_segmentation.process(rgb_frame)

    # Create a mask from the segmentation results
    condition = segmentation_results.segmentation_mask > 0.5

    # Create a green background
    green_bg = np.zeros_like(frame)
    green_bg[:] = (0, 255, 0)  # Green color in BGR

    # Combine the frame with the green background using the mask
    frame = np.where(condition[:, :, None], frame, green_bg)

    # Check if hand landmarks are found
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks and connections
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the positions of all the fingertips
            fingertip_positions = [
                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            ]

            # Convert positions to pixel coordinates
            fingertip_coords = [(int(tip.x * frame.shape[1]), int(tip.y * frame.shape[0])) for tip in fingertip_positions]

            # Calculate the distances between the thumb tip and other fingertips
            thumb_tip = fingertip_coords[0]
            distances = [calculate_distance(thumb_tip, fingertip_coords[i]) for i in range(1, len(fingertip_coords))]

            # If all distances are below a certain threshold, clear the drawing points
            if all(distance < 40 for distance in distances):  # Adjust threshold for grip detection
                drawing_points.clear()  # Clear the drawing points

            # Get the thumb and index finger tip positions
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert positions to pixel coordinates
            thumb_tip_x, thumb_tip_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            index_tip_x, index_tip_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

            # Calculate the distance between thumb and index finger tips
            pinch_distance = calculate_distance((thumb_tip_x, thumb_tip_y), (index_tip_x, index_tip_y))

            # If pinch distance is small, we assume a pinch gesture
            if pinch_distance < 10:  # Adjust threshold for pinch detection
                # Add the pinch position to the drawing points list
                drawing_points.append((index_tip_x, index_tip_y))

    # Draw the path
    if drawing_points:
        for i in range(1, len(drawing_points)):
            cv2.line(frame, drawing_points[i - 1], drawing_points[i], (255, 255, 255), 5)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Limit FPS to 60
    if fps > 60:
        time.sleep(1 / 60 - (curr_time - prev_time))

    # Display FPS on the frame
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Hand Tracking with Drawing', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()