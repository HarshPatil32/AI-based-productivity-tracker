import { api } from "./client"
import type { Session, CreateSessionPayload } from "@/types/session"

export const getSessions = (userId: string) =>
  api.get<Session[]>(`/sessions/user/${userId}`).then(r => r.data)

export const getSession = (id: string) =>
  api.get<Session>(`/sessions/${id}`).then(r => r.data)

export const createSession = (payload: CreateSessionPayload) =>
  api.post<Session>("/sessions", payload).then(r => r.data)

export const listSessions = (params?: { page?: number; page_size?: number }) =>
  api.get<Session[]>("/sessions", { params }).then(r => r.data)

export const endSession = (id: string) =>
  api.post<Session>(`/sessions/${id}/end`).then(r => r.data)

export const deleteSession = (id: string) =>
  api.delete(`/sessions/${id}`).then(r => r.data)
