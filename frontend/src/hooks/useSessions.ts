import { useEffect, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  createSession,
  deleteSession,
  endSession,
  getSession,
  getSessionSummary,
  listSessions,
  SESSIONS_PAGE_SIZE,
} from '../api/sessions';
import type { CreateSessionPayload } from '../types/session';

export function useSessions() {
  const queryClient = useQueryClient();
  const [page, setPage] = useState(1);
  const [maxUnlockedPage, setMaxUnlockedPage] = useState(1);

  const sessionsQuery = useQuery({
    queryKey: ['sessions', page],
    queryFn: () => listSessions({ limit: SESSIONS_PAGE_SIZE, offset: (page - 1) * SESSIONS_PAGE_SIZE }),
  });

  useEffect(() => {
    if ((sessionsQuery.data?.length ?? 0) === SESSIONS_PAGE_SIZE) {
      setMaxUnlockedPage((prev) => Math.max(prev, page + 1));
    }
  }, [sessionsQuery.data, page]);

  const createMutation = useMutation({
    mutationFn: (payload: CreateSessionPayload) => createSession(payload),
    onSuccess: () => {
      setPage(1);
      setMaxUnlockedPage(1);
      queryClient.invalidateQueries({ queryKey: ['sessions'] });
    },
  });

  const endMutation = useMutation({
    mutationFn: (id: string) => endSession(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['sessions'] }),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteSession(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['sessions'] }),
  });

  return {
    sessions: sessionsQuery.data ?? [],
    isLoading: sessionsQuery.isLoading,
    isFetching: sessionsQuery.isFetching,
    page,
    setPage,
    maxUnlockedPage,
    error: sessionsQuery.error,
    create: createMutation,
    end: endMutation,
    remove: deleteMutation,
  };
}

export function useSessionSummary() {
  const result = useQuery({
    queryKey: ['sessions', 'summary'],
    queryFn: () => getSessionSummary(),
  });
  return {
    summary: result.data,
    isLoading: result.isLoading,
    error: result.error,
  };
}

export function useSession(id: string | undefined) {
  return useQuery({
    queryKey: ['sessions', id],
    queryFn: () => getSession(id!),
    enabled: !!id,
  });
}
