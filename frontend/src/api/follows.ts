import apiClient from './client';
import type { FollowRelation } from '../types/feed';
import type { User } from '../types/auth';

export const follow = (followeeId: string) =>
  apiClient
    .post<FollowRelation>(`/users/${followeeId}/follow`)
    .then((r) => r.data);

export const unfollow = (followeeId: string) =>
  apiClient.delete(`/users/${followeeId}/follow`).then((r) => r.data);

export const getFollowers = (userId: string, limit = 20, offset = 0) =>
  apiClient
    .get<User[]>(`/users/${userId}/followers`, { params: { limit, offset } })
    .then((r) => r.data);

export const getFollowing = (userId: string, limit = 20, offset = 0) =>
  apiClient
    .get<User[]>(`/users/${userId}/following`, { params: { limit, offset } })
    .then((r) => r.data);
