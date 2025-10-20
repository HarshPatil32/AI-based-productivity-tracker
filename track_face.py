import cv2
import dlib
import numpy as np
import time
import logging


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    """Calculate the Eye Aspect Ratio (EAR) to detect blinks."""
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))  
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))  
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))  
    return (A + B) / (2.0 * C)

# 3D model points of facial landmarks for head pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (landmark 30)
    (0.0, -330.0, -65.0),        # Chin (landmark 8)
    (-225.0, 170.0, -135.0),     # Left eye corner (landmark 36)
    (225.0, 170.0, -135.0),      # Right eye corner (landmark 45)
    (-150.0, -150.0, -125.0),    # Left mouth corner (landmark 48)
    (150.0, -150.0, -125.0)      # Right mouth corner (landmark 54)
], dtype=np.float64)

def detect_attention():
    logging.basicConfig(filename="attention_tracker_log.txt", level=logging.DEBUG, format='%(asctime)s - %(message)s', filemode='w')
    cap = cv2.VideoCapture(2)
    EAR_THRESHOLD = 0.2  
    closed_duration = 0   
    last_closed_time = None  
    logging_interval = 2  
    HEAD_YAW_THRESHOLD = 30  # Threshold for detecting if the user is looking away, adjustable but worked best
    HEAD_PITCH_THRESHOLD = 25  # Threshold for detecting if looking up or down, also adjustable

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
            right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            eye_state = "Open" if avg_ear > EAR_THRESHOLD else "Closed"
            color = (0, 255, 0) if eye_state == "Open" else (0, 0, 255)

            for (x, y) in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, color, -1)

            cv2.putText(frame, f"Eye State: {eye_state}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if eye_state == "Closed":
                if last_closed_time is None:
                    last_closed_time = time.time()
                else:
                    closed_duration = time.time() - last_closed_time
                
                if closed_duration >= logging_interval:
                    logging.info(f'User wasn\'t paying attention for {closed_duration:.2f} seconds (eyes closed)')
                    last_closed_time = time.time()
            else:
                last_closed_time = None

            image_points = np.array([
                (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
                (landmarks.part(8).x, landmarks.part(8).y),    # Chin
                (landmarks.part(36).x, landmarks.part(36).y),  # Left eye corner
                (landmarks.part(45).x, landmarks.part(45).y),  # Right eye corner
                (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth corner
                (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
            ], dtype=np.float64)

            # Camera matrix (assuming a simple pinhole camera model), this is a little finicky, try with * 1.5 maybe
            focal_length = frame.shape[1]
            center = (frame.shape[1] / 2, frame.shape[0] / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)

            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

            # SolvePnP to estimate rotation vector and translation vector
            success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
            if success:
                rvec_matrix, _ = cv2.Rodrigues(rotation_vector) # This converts it into a 3x3 rotation matrix
                # Combines rotation and translation into one structure
                proj_matrix = np.hstack((rvec_matrix, np.zeros((3, 1)))) # Adds a zero translation column to the matrix, makes it 3x4
                _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(proj_matrix) # Angles has all the stuff i need
                yaw, pitch, _ = angles.flatten() # We need a flat list

                yaw = yaw - 145 # This is an offset bc the values were wack before
                
                # Check if user is facing away
                if abs(yaw) > HEAD_YAW_THRESHOLD or abs(pitch) > HEAD_PITCH_THRESHOLD:
                    logging.info(f'User is not facing the screen (Yaw: {yaw:.2f}, Pitch: {pitch:.2f})')
                    cv2.putText(frame, "Not Paying Attention!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(frame, f"Yaw: {yaw:.2f}, Pitch: {pitch:.2f}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Face & Attention Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect_attention()