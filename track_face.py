import cv2
import dlib
import numpy as np
import time
import logging



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))  
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))  
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))  
    return (A + B) / (2.0 * C)

def detect_eye_state():
    logging.basicConfig(filename="attention_tracker_log.txt", level=logging.DEBUG, format='%(asctime)s - %(message)s', filemode = 'w')
    cap = cv2.VideoCapture(0)
    EAR_THRESHOLD = 0.2 
    closed_duration = 0 # Check how long eyes are closed
    last_closed_time = None # What time were the eyes last closed
    logging_interval = 2 # Log after 2 seconds of closed eyes

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  

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
                logging.info(f"CLOSED for {closed_duration}")
                if last_closed_time is None: # Check if this is the first time the eyes are closed
                    last_closed_time = time.time()
                else:
                    closed_duration = time.time() - last_closed_time # See how long it has been since they were last closed
                
                if closed_duration>= logging_interval:
                    logging.info(f'User wasn\'t paying attention for {closed_duration:.2f} seconds, they had their eyes closed')
                    last_closed_time = time.time() # Reset the time
            else:
                last_closed_time = None

        cv2.imshow("Face & Eye Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect_eye_state()
