import { api } from "./client"
import type { AuthResponse, User } from "@/types/auth"

export const register = (data: { email: string; username: string; password: string }) =>
  api.post<AuthResponse>("/auth/register", data).then(r => r.data)

export const login = (data: { email: string; password: string }) =>
  api.post<AuthResponse>("/auth/login", data).then(r => r.data)

export const logout = () => api.post("/auth/logout")

export const getMe = () =>
  api.get<User>("/auth/me").then(r => r.data)
