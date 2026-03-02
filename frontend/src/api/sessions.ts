import apiClient from './client';
import type { Session, CreateSessionPayload } from '../types/session';

export const createSession = (payload: CreateSessionPayload) =>
  apiClient.post<Session>('/sessions', payload).then((r) => r.data);

export const getSession = (id: number) =>
  apiClient.get<Session>(`/sessions/${id}`).then((r) => r.data);

export const listSessions = (params?: { page?: number; page_size?: number }) =>
  apiClient
    .get<Session[]>('/sessions', { params })
    .then((r) => r.data);

export const endSession = (id: number) =>
  apiClient.post<Session>(`/sessions/${id}/end`).then((r) => r.data);

export const deleteSession = (id: number) =>
  apiClient.delete(`/sessions/${id}`).then((r) => r.data);
