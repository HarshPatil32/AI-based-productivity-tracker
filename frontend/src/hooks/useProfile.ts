import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { getUserProfile, updateProfile } from '../api/users';
import { follow, unfollow, getFollowers, getFollowing } from '../api/follows';
import type { UpdateProfilePayload } from '../api/users';

export function useProfile(username: string) {
  const queryClient = useQueryClient();

  const profileQuery = useQuery({
    queryKey: ['users', username],
    queryFn: () => getUserProfile(username),
    enabled: !!username,
  });

  const followersQuery = useQuery({
    queryKey: ['users', username, 'followers'],
    queryFn: () => getFollowers(username),
    enabled: !!username,
  });

  const followingQuery = useQuery({
    queryKey: ['users', username, 'following'],
    queryFn: () => getFollowing(username),
    enabled: !!username,
  });

  const updateMutation = useMutation({
    mutationFn: (payload: UpdateProfilePayload) => updateProfile(payload),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ['users', username] }),
  });

  const followMutation = useMutation({
    mutationFn: (id: number) => follow(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users', username, 'followers'] });
    },
  });

  const unfollowMutation = useMutation({
    mutationFn: (id: number) => unfollow(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users', username, 'followers'] });
    },
  });

  return {
    profile: profileQuery.data,
    isLoading: profileQuery.isLoading,
    followers: followersQuery.data ?? [],
    following: followingQuery.data ?? [],
    update: updateMutation,
    follow: followMutation,
    unfollow: unfollowMutation,
  };
}
