import cv2
import mediapipe as mp
import numpy as np
import time
import math
import pygame
import random

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

# Initialize pygame
pygame.init()

# Set up the screen
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Catch the Falling Objects")

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# Basket properties
basket_width = 100
basket_height = 20
basket_x = 350
basket_y = 550

# Object properties
object_width = 20
object_height = 20
object_x = random.randint(0, 780)
object_y = 0
object_speed = 20

# Score
score = 0
font = pygame.font.Font(None, 36)

# Create a list to store drawing points
drawing_points = []

# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

# Game loop
running = True
while running and cap.isOpened():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

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

    # Check if hand landmarks are found
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the thumb and index finger tip positions
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert positions to pixel coordinates
            thumb_tip_x, thumb_tip_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            index_tip_x, index_tip_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

            # Calculate the distance between thumb and index finger tips
            pinch_distance = calculate_distance((thumb_tip_x, thumb_tip_y), (index_tip_x, index_tip_y))

            # Calculate the midpoint between thumb and index finger tips
            midpoint_x = (thumb_tip_x + index_tip_x) // 2
            midpoint_y = (thumb_tip_y + index_tip_y) // 2

            # Map the midpoint to screen coordinates
            screen_x = np.interp(midpoint_x, [0, frame.shape[1]], [0, 800])
            screen_y = np.interp(midpoint_y, [0, frame.shape[0]], [0, 600])

            # Move the basket with the hand tracking
            basket_x = screen_x - basket_width // 2

    # Move the object down
    object_y += object_speed

    # Check if the object is caught by the basket
    if (basket_x < object_x < basket_x + basket_width or basket_x < object_x + object_width < basket_x + basket_width) and basket_y < object_y + object_height < basket_y + basket_height:
        score += 1
        object_x = random.randint(0, 780)
        object_y = 0

    # Check if the object falls off the screen
    if object_y > 600:
        object_x = random.randint(0, 780)
        object_y = 0

    # Clear the screen
    screen.fill(WHITE)

    # Draw the basket
    pygame.draw.rect(screen, BLUE, (basket_x, basket_y, basket_width, basket_height))

    # Draw the falling object
    pygame.draw.rect(screen, BLACK, (object_x, object_y, object_width, object_height))

    # Draw the score
    score_text = font.render(f"Score: {score}", True, (0, 0, 0))
    screen.blit(score_text, (10, 10))

    # Update the display
    pygame.display.flip()

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Limit FPS to 60
    if fps > 60:
        time.sleep(1 / 60 - (curr_time - prev_time))

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Quit pygame
pygame.quit()