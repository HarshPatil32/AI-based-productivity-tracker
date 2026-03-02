export interface Session {
  id: number;
  user_id: number;
  title: string;
  description?: string;
  start_time: string;
  end_time?: string;
  duration_seconds?: number;
  attention_score?: number;
  focus_score?: number;
  status: 'active' | 'completed' | 'paused';
  created_at: string;
}

export interface CreateSessionPayload {
  title: string;
  description?: string;
}

export interface SessionStats {
  total_sessions: number;
  total_duration_seconds: number;
  average_attention_score: number;
  average_focus_score: number;
}
