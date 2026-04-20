// ----- Request payload (mirrors backend SessionCreate) -----

export interface CreateSessionPayload {
  started_at: string;
  ended_at: string;
  duration_seconds: number;
  eyes_closed_time: number;
  face_missing_time: number;
  head_pose_off_time: number;
  total_attention_lost: number;
  notes?: string;
}

// ----- Response model (mirrors backend SessionResponse) -----

export interface Session extends CreateSessionPayload {
  id: string;           // UUID
  user_id: string;      // UUID
  attention_score: number;
  focus_score: number;
  created_at: string;
}

// ----- Summary (mirrors backend SessionSummary) -----

export interface SessionSummary {
  total_sessions: number;
  total_study_seconds: number;
  avg_attention_score: number;
  avg_focus_score: number;
  avg_eyes_closed_time: number;
  avg_face_missing_time: number;
  avg_head_pose_off_time: number;
  total_attention_lost: number;
}
