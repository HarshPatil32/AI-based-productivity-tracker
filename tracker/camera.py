import cv2

def get_video_capture(index=0):
    return cv2.VideoCapture(index)

def release_resources(cap):
    cap.release()
    cv2.destroyAllWindows()
