// ----- Request payloads -----

export interface RegisterPayload {
  email: string;
  password: string;
  username: string;
  full_name?: string;
}

export interface LoginPayload {
  email: string;
  password: string;
}

export interface PasswordUpdatePayload {
  current_password: string;
  new_password: string;
}

export interface TokenRefreshPayload {
  refresh_token: string;
}

// ----- Response models (mirrors backend UserResponse / TokenResponse) -----

export interface User {
  id: string;           // UUID
  email: string;
  username: string;
  full_name?: string;
  bio?: string;
  avatar_url?: string;
  created_at: string;
  updated_at?: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;   // "bearer"
  expires_in: number;   // seconds
  user: User;
}

// Alias used by the API layer
export type AuthResponse = TokenResponse;
