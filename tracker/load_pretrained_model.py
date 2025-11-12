from ultralytics import YOLO
import cv2
import numpy as np


model = YOLO("yolov8n.pt")

video_path = 'wave.MOV'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


output_path = "output_video.mov"
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break  

    
    results = model(frame)

    
    for result in results:
        boxes = result.boxes.xyxy  
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  

    
    out.write(frame)


cap.release()
out.release()
cv2.destroyAllWindows()




