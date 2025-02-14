import cv2
import logging



log_file = 'face_detection_log.txt'
with open(log_file, 'w'):
    pass
logging.basicConfig(filename=log_file, level = logging.INFO,
                    format = '%(asctime)s - %(message)s')

# Load OpenCV's pre-trained Haar cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_face():
    cap = cv2.VideoCapture(0)
    logging.info("Started face detection")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        faces = face_cascade.detectMultiScale(gray, 1.05, 20)  
        
        logging.info(f"Faces detected: {len(faces)}")
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
            roi_gray = gray[y:y+w, x:x+w]
            roi_color = frame[y:y+h, x:x+h]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 20)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)

        cv2.imshow("Face Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



import tkinter as tk

root = tk.Tk()
root.title("AI Productivity Tracker")

btn = tk.Button(root, text="Start Face Detection", command=detect_face)
btn.pack()

root.mainloop()
