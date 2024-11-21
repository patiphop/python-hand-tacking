from flask import Flask, request, Response
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)
CORS(app)

mp_hands = mp.solutions.hands
mp_selfie_segmentation = mp.solutions.selfie_segmentation

hands = mp_hands.Hands()
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation()

@app.route('/process', methods=['POST'])
def process_frame():
    nparr = np.frombuffer(request.data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return Response("Failed to decode image", status=400)

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        return Response(f"Error in cvtColor: {e}", status=500)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Process the frame for segmentation
    segmentation_results = selfie_segmentation.process(rgb_frame)

    # Create a mask from the segmentation results
    condition = segmentation_results.segmentation_mask > 0.5

    # Combine the frame with the green background using the mask
    green_bg = np.zeros_like(frame)
    green_bg[:] = (0, 255, 0)  # Green color in BGR
    frame = np.where(condition[:, :, None], frame, green_bg)

    _, buffer = cv2.imencode('.jpg', frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)