import cv2
import numpy as np
import tensorflow as tf
import time

latest_frame = None
posture_state = None

interpreter = tf.lite.Interpreter(model_path="models/movenet.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_frame(frame):
    """
    Resize and normalize the frame for MoveNet input
    - Resize to 192x192 which is what MoveNet expects
    - Convert BGF to RGB (again what MoveNet expects)
    - Normalize pixel values to 0 and 1
    - Add batch dimension
    """
    image = cv2.resize(frame, (192, 192))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)


def draw_keypoints(frame, keypoints, confidence_threshold=0.3):
    """
    Draws keypoints on the frame
    - Convert the coordinates to pixels
    - Mark them with green dots (temporary for testing pruposes)
    """
    h, w, _ = frame.shape
    for kp in keypoints[0,0]:
        y, x, confidence = kp[0], kp[1], kp[2]
        if confidence > confidence_threshold:
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)


def analyze_posture(frame):
    """
    Runs pose estimation and returns the state of the postire and annotated frame
    - Just returns how good the posture is
    """
    print('Entered analyze posture')
    input_image = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])

        
    # Get the keypoints we care about: nose, left shoulder, right shoulder 
    keypoints = keypoints[0,0]
    nose = keypoints[0]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    posture = "Proper"

    # Check how for the nose is from the center of the shoulders
    shoulder_center_x = (left_shoulder[1] + right_shoulder[1]) / 2
    nose_offset = nose[1] - shoulder_center_x
    
    # Find the vertical drop of the nose
    shoulder_y_avg = (left_shoulder[0] + right_shoulder[0]) / 2
    head_drop =  nose[0] - shoulder_y_avg

    if abs(nose_offset) > 0.05:
        posture = "Leaning"
    elif head_drop > 0.05:
        posture = "Slouch"

    draw_keypoints(frame, np.expand_dims(keypoints, axis=(0, 1)))

    cv2.putText(frame, f"Posture: {posture}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return frame, posture


# For logging posture time durations
posture_timer = {
    "Proper": 0.0,
    "Slouch": 0.0,
    "Leaning": 0.0
}
last_posture_time = None
def posture_thread():
    global latest_frame, posture_state, last_posture_time

    while True:
        if latest_frame is not None:
            frame_copy = latest_frame.copy()

            _, current_posture = analyze_posture(frame_copy)
            posture_state = current_posture

            # Track the time
            now = time.time()
            if last_posture_time is None:
                last_posture_time = now
            else:
                duration = now - last_posture_time
                posture_timer[current_posture] += duration
                last_posture_time = now

        time.sleep(0.2) # Gonna run it at 5 fps for now so less CPU usage

