import { api } from "./client"
import type { Session, CreateSessionPayload, SessionSummary } from "@/types/session"

export const SESSIONS_PAGE_SIZE = 20;

export const getSessions = (userId: string) =>
  api.get<Session[]>(`/sessions/user/${userId}`).then(r => r.data)

export const getSession = (id: string) =>
  api.get<Session>(`/sessions/${id}`).then(r => r.data)

export const createSession = (payload: CreateSessionPayload) =>
  api.post<Session>("/sessions", payload).then(r => r.data)

export const listSessions = (params?: { limit?: number; offset?: number }) =>
  api.get<Session[]>("/sessions/me", { params }).then(r => r.data)

export const endSession = (id: string) =>
  api.post<Session>(`/sessions/${id}/end`).then(r => r.data)

export const deleteSession = (id: string) =>
  api.delete(`/sessions/${id}`).then(r => r.data)

export const getSessionSummary = () =>
  api.get<SessionSummary>("/sessions/me/summary").then(r => r.data)
