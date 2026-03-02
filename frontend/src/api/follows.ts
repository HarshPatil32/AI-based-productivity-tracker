import apiClient from './client';
import type { FollowRelation } from '../types/feed';
import type { User } from '../types/auth';

export const follow = (followeeId: number) =>
  apiClient
    .post<FollowRelation>(`/follows/${followeeId}`)
    .then((r) => r.data);

export const unfollow = (followeeId: number) =>
  apiClient.delete(`/follows/${followeeId}`).then((r) => r.data);

export const getFollowers = (username: string) =>
  apiClient
    .get<User[]>(`/users/${username}/followers`)
    .then((r) => r.data);

export const getFollowing = (username: string) =>
  apiClient
    .get<User[]>(`/users/${username}/following`)
    .then((r) => r.data);
