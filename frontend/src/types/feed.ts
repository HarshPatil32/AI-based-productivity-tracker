import type { Session } from './session';
import type { User } from './auth';

export interface FeedItem {
  id: number;
  session: Session;
  author: User;
  created_at: string;
}

export interface FeedResponse {
  items: FeedItem[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
}

export interface FollowPayload {
  followee_id: number;
}

export interface FollowRelation {
  follower_id: number;
  followee_id: number;
  created_at: string;
}
