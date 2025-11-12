from attention import detect_attention
import time

def calculate_productivity_metrics(attention_metrics, session_duration):
    """Calculate productivity metrics based on attention tracking"""
    
    attention_lost = attention_metrics['total_attention_lost']
    work_time = session_duration - attention_lost
    work_time = max(0, work_time)
    
    focus_score = (work_time / session_duration * 100) if session_duration > 0 else 0
    attention_score = focus_score  # Same as focus score now
    
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
        'distracted_time': attention_lost,
        'attention_lost': attention_lost,
        'focus_score': focus_score,
        'attention_score': attention_score,
        'quality': quality
    }

if __name__ == "__main__":
    # Record session start time
    session_start_time = time.time()
    print("Starting productivity tracking session...")
    print("Press 'q' in the camera window to stop tracking")

    attention_metrics = detect_attention()
    
    session_end_time = time.time()
    session_duration = session_end_time - session_start_time
    
    productivity = calculate_productivity_metrics(attention_metrics, session_duration)
    
    print("\n========= PRODUCTIVITY METRICS =========")
    print(f"Total session time: {productivity['session_duration']:.1f} seconds ({productivity['session_duration']/60:.1f} minutes)")
    print(f"Focused time: {productivity['work_time']:.1f} seconds ({productivity['focus_score']:.1f}%)")
    print(f"Distracted time: {productivity['distracted_time']:.1f} seconds ({100-productivity['focus_score']:.1f}%)")
    
    print("\nDetailed Breakdown:")
    attention_percentage = (productivity['attention_lost'] / session_duration * 100) if session_duration > 0 else 0
    print(f"- Attention lost: {productivity['attention_lost']:.1f} seconds ({attention_percentage:.1f}%)")
    
    print("\nScores:")
    print(f"Focus Score: {productivity['focus_score']:.1f}%")
    print(f"Attention Score: {productivity['attention_score']:.1f}%")
    print(f"Session Quality: {productivity['quality']}")
    
    print("\n========= SESSION COMPLETE =========")
