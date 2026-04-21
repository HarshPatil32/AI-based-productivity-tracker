import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { getUserProfile, updateProfile } from '../api/users';
import { follow, unfollow, getFollowers, getFollowing } from '../api/follows';
import { getSessions } from '../api/sessions';
import type { UpdateProfilePayload } from '../api/users';

export function useProfile(username: string) {
  const queryClient = useQueryClient();

  const profileQuery = useQuery({
    queryKey: ['users', username],
    queryFn: () => getUserProfile(username),
    enabled: !!username,
  });

  const profileId = profileQuery.data?.id;

  const followersQuery = useQuery({
    queryKey: ['users', username, 'followers'],
    queryFn: () => getFollowers(profileId!),
    enabled: !!profileId,
  });

  const followingQuery = useQuery({
    queryKey: ['users', username, 'following'],
    queryFn: () => getFollowing(profileId!),
    enabled: !!profileId,
  });

  const sessionsQuery = useQuery({
    queryKey: ['users', username, 'sessions'],
    queryFn: () => getSessions(profileId!),
    enabled: !!profileId,
  });

  const updateMutation = useMutation({
    mutationFn: (payload: UpdateProfilePayload) => updateProfile(payload),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ['users', username] }),
  });

  const followMutation = useMutation({
    mutationFn: (id: string) => follow(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users', username, 'followers'] });
    },
  });

  const unfollowMutation = useMutation({
    mutationFn: (id: string) => unfollow(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users', username, 'followers'] });
    },
  });

  return {
    profile: profileQuery.data,
    isLoading: profileQuery.isLoading,
    error: profileQuery.error,
    followers: followersQuery.data ?? [],
    following: followingQuery.data ?? [],
    profileSessions: sessionsQuery.data ?? [],
    isLoadingSessions: sessionsQuery.isLoading,
    update: updateMutation,
    follow: followMutation,
    unfollow: unfollowMutation,
  };
}
