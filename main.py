from attention import detect_attention
from posture import posture_thread, posture_timer
import threading
import time

if __name__ == "__main__":

    # Start the posture detection thread
    posture_t = threading.Thread(target=posture_thread, daemon=True)
    posture_t.start()


    detect_attention()

    total_posture_time = sum(posture_timer.values())
    with open("attention_tracker_log.txt", "a") as f:
        f.write("\n========= POSTURE SUMMARY =========\n")
        for posture, duration in posture_timer.items():
            f.write(f"{posture} time: {duration:.2f} sec\n")
        f.write(f"Total tracked posture time: {total_posture_time:.2f} sec\n")
