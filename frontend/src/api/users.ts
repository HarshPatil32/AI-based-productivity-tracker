import apiClient from './client';
import type { User } from '../types/auth';

export interface UpdateProfilePayload {
  full_name?: string;
  bio?: string;
  avatar_url?: string;
}

export const getUserProfile = (username: string) =>
  apiClient.get<User>(`/users/${username}`).then((r) => r.data);

export const updateProfile = (payload: UpdateProfilePayload) =>
  apiClient.put<User>('/users/me', payload).then((r) => r.data);

export const searchUsers = (query: string) =>
  apiClient
    .get<User[]>('/users/search', { params: { q: query } })
    .then((r) => r.data);
