import cv2
import time
import numpy as np
import logging
from face_utils import get_landmarks, eye_aspect_ratio
from config import EAR_THRESHOLD, HEAD_YAW_THRESHOLD, HEAD_PITCH_THRESHOLD, model_points
from camera import get_video_capture, release_resources
import posture


def detect_attention():
    print("Starting detect_attention function")
    try:
        logging.basicConfig(
            filename="attention_tracker_log.txt",
            level=logging.DEBUG,
            format='%(asctime)s - %(message)s'
        )
        logging.info("Starting attention detection")
        print("Logging initialized")

        cap = get_video_capture(1)
        logging.info("Video capture initialized")
        print("Video capture initialized")
        
        closed_duration = 0
        last_closed_time = None
        logging_interval = 2

        missing_face_duration = 0
        last_missing_time = None

        head_pose_duration = 0
        last_pose_off_time = None

        print("Starting main loop")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.info("Failed to read frame, breaking loop")
                print("Failed to read frame, breaking loop")
                break

            posture.set_latest_frame(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            landmarks_data = get_landmarks(gray)

            # Handle missing face detection
            if not landmarks_data:
                cv2.putText(frame, "No Face Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if last_missing_time is None:
                    last_missing_time = time.time()
                else:
                    missing_face_duration += time.time() - last_missing_time
                    last_missing_time = time.time()
                    logging.info(f"Face missing for {missing_face_duration:.2f} seconds")
            else:
                last_missing_time = None

            for face, landmarks in landmarks_data:
                try:
                    # Extract eye coordinates
                    left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
                    right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]

                    # Compute EAR
                    avg_ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                    eye_state = "Open" if avg_ear > EAR_THRESHOLD else "Closed"
                    color = (0, 255, 0) if eye_state == "Open" else (0, 0, 255)

                    for (x, y) in left_eye + right_eye:
                        cv2.circle(frame, (x, y), 2, color, -1)

                    cv2.putText(frame, f"Eye State: {eye_state}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    if eye_state == "Closed":
                        print("Entered eyses closed")
                        if last_closed_time is None:
                            last_closed_time = time.time()
                        else:
                            print("Entered eyes else")
                            closed_duration = time.time() - last_closed_time
                            if closed_duration >= logging_interval:
                                print(f"Eyes closed for {closed_duration:.2f} seconds")
                                last_closed_time = time.time()
                    else:
                        last_closed_time = None

                    # Head Pose Estimation
                    image_points = np.array([
                        (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
                        (landmarks.part(8).x, landmarks.part(8).y),    # Chin
                        (landmarks.part(36).x, landmarks.part(36).y),  # Left eye corner
                        (landmarks.part(45).x, landmarks.part(45).y),  # Right eye corner
                        (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth corner
                        (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
                    ], dtype=np.float64)

                    focal_length = frame.shape[1]
                    center = (frame.shape[1] / 2, frame.shape[0] / 2)
                    camera_matrix = np.array([
                        [focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]
                    ], dtype=np.float64)

                    dist_coeffs = np.zeros((4, 1))  # No distortion
                    success, rotation_vector, _ = cv2.solvePnP(
                        model_points, image_points, camera_matrix, dist_coeffs
                    )

                    if success:
                        rvec_matrix, _ = cv2.Rodrigues(rotation_vector)
                        proj_matrix = np.hstack((rvec_matrix, np.zeros((3, 1))))
                        _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(proj_matrix)
                        yaw, pitch, _ = angles.flatten()

                        if abs(yaw) < HEAD_YAW_THRESHOLD or abs(pitch) > HEAD_PITCH_THRESHOLD:
                            print("Entered here yaw stuff")
                            cv2.putText(frame, "Not Paying Attention!", (50, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            if last_pose_off_time is None:
                                last_pose_off_time = time.time()
                            else:
                                head_pose_duration = time.time() - last_pose_off_time
                                last_pose_off_time = time.time()

                            print(f"Head turned away (Yaw: {abs(yaw):.2f} its supposed to be {HEAD_YAW_THRESHOLD}, Pitch: {pitch:.2f}), its supposed to be {HEAD_PITCH_THRESHOLD}")
                        else:
                            last_pose_off_time = None
                            
                    
                except Exception as e:
                    print(f"Error processing face: {e}")
                    logging.error(f"Error processing face: {e}", exc_info=True)

            cv2.imshow("Face & Attention Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("Exited main loop, writing session summary")
        logging.info("Attention detection loop ended")
        logging.info("========= SESSION SUMMARY =========")
        logging.info(f"Total time with eyes closed (≥{logging_interval}s chunks): {closed_duration:.2f} sec")
        logging.info(f"Total time face was missing: {missing_face_duration:.2f} sec")
        logging.info(f"Total time head pose was off: {head_pose_duration:.2f} sec")

        total_attention_lost = closed_duration + missing_face_duration + head_pose_duration
        logging.info(f"==> Total 'not paying attention' time: {total_attention_lost:.2f} sec")

        release_resources(cap)
        print("Session summary written, resources released")
        
    except Exception as e:
        logging.error(f"Error in attention detection: {e}", exc_info=True)
        print(f"Attention detection error: {e}")
        if 'cap' in locals():
            release_resources(cap)
