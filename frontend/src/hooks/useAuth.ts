import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { login, logout, register, getMe } from '../api/auth';
import { useAuthStore } from '../store/authStore';
import type { LoginPayload, RegisterPayload } from '../types/auth';

export function useAuth() {
  const { user, token, setAuth, clearAuth } = useAuthStore();
  const isAuthenticated = !!token;
  const queryClient = useQueryClient();

  const meQuery = useQuery({
    queryKey: ['me'],
    queryFn: getMe,
    enabled: isAuthenticated,
    staleTime: 5 * 60 * 1000,
  });

  const loginMutation = useMutation({
    mutationFn: (payload: LoginPayload) => login(payload),
    onSuccess: (data) => {
      if (data.user) {
        setAuth(data.user, data.access_token);
      }
      queryClient.invalidateQueries({ queryKey: ['me'] });
    },
  });

  const registerMutation = useMutation({
    mutationFn: (payload: RegisterPayload) => register(payload),
  });

  const logoutMutation = useMutation({
    mutationFn: logout,
    onSettled: () => {
      clearAuth();
      queryClient.clear();
    },
  });

  // Keep store user in sync with server
  if (meQuery.data && meQuery.data !== user) {
    setAuth(meQuery.data, token!);
  }

  return {
    user: meQuery.data ?? user,
    isAuthenticated,
    isLoading: meQuery.isLoading,
    login: loginMutation,
    register: registerMutation,
    logout: logoutMutation,
  };
}
