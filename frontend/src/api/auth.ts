import apiClient from './client';
import type { AuthTokens, RegisterPayload, User } from '../types/auth';

export const register = (payload: RegisterPayload) =>
  apiClient.post<User>('/auth/register', payload).then((r) => r.data);

export const login = (payload: { username: string; password: string }) => {
  const form = new URLSearchParams();
  form.append('username', payload.username);
  form.append('password', payload.password);
  return apiClient
    .post<AuthTokens>('/auth/login', form, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    })
    .then((r) => r.data);
};

export const logout = () =>
  apiClient.post('/auth/logout').then((r) => r.data);

export const refreshToken = () =>
  apiClient.post<AuthTokens>('/auth/refresh').then((r) => r.data);

export const getMe = () =>
  apiClient.get<User>('/auth/me').then((r) => r.data);
