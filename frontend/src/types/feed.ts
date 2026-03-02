// ----- Feed session card (mirrors backend FeedSessionResponse) -----
// A single flat object combining session data + author info.

export interface FeedSession {
  id: string;                   // UUID
  user_id: string;              // UUID
  username: string;
  full_name?: string;
  avatar_url?: string;

  title: string;
  description?: string;
  session_duration: number;     // seconds
  focused_time: number;
  focus_score: number;
  attention_score: number;
  quality: string;              // e.g. "Excellent" | "Good" | "Fair" | "Poor"
  session_date: string;
  session_start_time?: string;
  session_end_time?: string;
  likes_count: number;
  comments_count: number;
  created_at: string;
}

// ----- Follow models (mirrors backend FollowResponse / FollowerEntry) -----

export interface FollowPayload {
  followee_id: string;          // UUID
}

export interface FollowRelation {
  follower_id: string;          // UUID
  following_id: string;         // UUID
  created_at: string;
}

export interface FollowerEntry {
  id: string;                   // UUID
  username: string;
  full_name?: string;
  avatar_url?: string;
  total_study_time: number;     // seconds
}
