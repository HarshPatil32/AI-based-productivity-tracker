from attention import detect_attention
from posture import posture_thread, posture_timer
import threading
import time

def calculate_productivity_metrics(attention_metrics, posture_metrics, session_duration):
    """Calculate comprehensive productivity metrics"""
    
    attention_lost = attention_metrics['total_attention_lost']
    slouch_time = posture_metrics.get('Slouch', 0)
    leaning_time = posture_metrics.get('Leaning', 0)
    proper_posture_time = posture_metrics.get('Proper', 0)
    
    poor_posture_time = slouch_time + leaning_time
    total_distracted_time = attention_lost + poor_posture_time
    work_time = session_duration - total_distracted_time
    
    work_time = max(0, work_time)
    
    focus_score = (work_time / session_duration * 100) if session_duration > 0 else 0
    attention_score = ((session_duration - attention_lost) / session_duration * 100) if session_duration > 0 else 0
    posture_score = (proper_posture_time / session_duration * 100) if session_duration > 0 else 0
    
    if focus_score >= 90:
        quality = "Excellent"
    elif focus_score >= 75:
        quality = "Good"
    elif focus_score >= 60:
        quality = "Fair"
    else:
        quality = "Poor"
    
    return {
        'session_duration': session_duration,
        'work_time': work_time,
        'distracted_time': total_distracted_time,
        'attention_lost': attention_lost,
        'poor_posture_time': poor_posture_time,
        'focus_score': focus_score,
        'attention_score': attention_score,
        'posture_score': posture_score,
        'quality': quality
    }

if __name__ == "__main__":
    # Record session start time
    session_start_time = time.time()
    print("Starting productivity tracking session...")
    
    # Start the posture detection thread
    posture_t = threading.Thread(target=posture_thread, daemon=True)
    posture_t.start()

    attention_metrics = detect_attention()
    
    session_end_time = time.time()
    session_duration = session_end_time - session_start_time
    
    total_posture_time = sum(posture_timer.values())
    
    print("\n========= POSTURE SUMMARY =========")
    for posture, duration in posture_timer.items():
        percentage = (duration / session_duration * 100) if session_duration > 0 else 0
        print(f"{posture} time: {duration:.2f} sec ({percentage:.1f}%)")
    print(f"Total tracked posture time: {total_posture_time:.2f} sec")
    
    productivity = calculate_productivity_metrics(attention_metrics, posture_timer, session_duration)
    
    print("\n========= PRODUCTIVITY METRICS =========")
    print(f"Total session time: {productivity['session_duration']:.1f} seconds ({productivity['session_duration']/60:.1f} minutes)")
    print(f"Work time: {productivity['work_time']:.1f} seconds ({productivity['focus_score']:.1f}%)")
    print(f"Distracted time: {productivity['distracted_time']:.1f} seconds ({100-productivity['focus_score']:.1f}%)")
    
    print("\nBreakdown:")
    attention_percentage = (productivity['attention_lost'] / session_duration * 100) if session_duration > 0 else 0
    posture_percentage = (productivity['poor_posture_time'] / session_duration * 100) if session_duration > 0 else 0
    print(f"- Attention issues: {productivity['attention_lost']:.1f} seconds ({attention_percentage:.1f}%)")
    print(f"- Poor posture: {productivity['poor_posture_time']:.1f} seconds ({posture_percentage:.1f}%)")
    
    print("\nScores:")
    print(f"Focus Score: {productivity['focus_score']:.1f}%")
    print(f"Attention Score: {productivity['attention_score']:.1f}%")
    print(f"Posture Score: {productivity['posture_score']:.1f}%")
    print(f"Session Quality: {productivity['quality']}")
    
    print("\n========= SESSION COMPLETE =========")
