import { createContext, useContext, useState } from "react"
import type { User } from "@/types/auth"

interface AuthState {
  user: User | null
  token: string | null
  setAuth: (user: User, token: string) => void
  clearAuth: () => void
}

export const AuthContext = createContext<AuthState>({} as AuthState)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(
    () => localStorage.getItem("access_token")
  )

  const setAuth = (u: User, t: string) => {
    setUser(u); setToken(t)
    localStorage.setItem("access_token", t)
  }
  const clearAuth = () => {
    setUser(null); setToken(null)
    localStorage.removeItem("access_token")
  }

  return (
    <AuthContext.Provider value={{ user, token, setAuth, clearAuth }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuthStore = () => useContext(AuthContext)
