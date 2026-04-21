import { useState, useRef } from 'react';
import { useParams } from 'react-router-dom';
import PageShell from '../components/layout/PageShell';
import UserAvatar from '../components/shared/UserAvatar';
import { SessionCard } from '../components/SessionCard';
import ErrorMessage from '../components/shared/ErrorMessage';
import { useProfile } from '../hooks/useProfile';
import { useAuth } from '../hooks/useAuth';
import type { User } from '../types/auth';

type SlimUser = Pick<User, 'id' | 'username' | 'full_name' | 'avatar_url'>;

import { useEffect, useRef, useState } from 'react';
function UserListModal({
  title,
  userId,
  fetchUsers,
  onClose,
}: {
  title: string;
  userId: string;
  fetchUsers: (userId: string, limit: number, offset: number) => Promise<SlimUser[]>;
  onClose: () => void;
}) {
  const PAGE_SIZE = 20;
  const [users, setUsers] = useState<SlimUser[]>([]);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(true);
  const listRef = useRef<HTMLUListElement>(null);

  const loadMore = async () => {
    if (loading || !hasMore) return;
    setLoading(true);
    setError(null);
    try {
      const newUsers = await fetchUsers(userId, PAGE_SIZE, offset);
      setUsers((prev) => [...prev, ...newUsers]);
      setOffset((prev) => prev + PAGE_SIZE);
      if (newUsers.length < PAGE_SIZE) setHasMore(false);
    } catch (e) {
      setError('Could not load users. Try again.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setUsers([]);
    setOffset(0);
    setHasMore(true);
    loadMore();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId]);

  // Infinite scroll
  useEffect(() => {
    const handleScroll = (e: Event) => {
      const el = listRef.current;
      if (!el || loading || !hasMore) return;
      if (el.scrollTop + el.clientHeight >= el.scrollHeight - 20) {
        loadMore();
      }
    };
    const el = listRef.current;
    if (el) el.addEventListener('scroll', handleScroll);
    return () => {
      if (el) el.removeEventListener('scroll', handleScroll);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loading, hasMore]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
      onClick={onClose}
    >
      <div
        className="bg-background rounded-xl border shadow-lg w-full max-w-sm max-h-[70vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-4 py-3 border-b">
          <h3 className="font-semibold text-sm">{title}</h3>
          <button
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground text-lg leading-none"
            aria-label="Close"
          >
            &times;
          </button>
        </div>
        <ul ref={listRef} className="overflow-y-auto divide-y" style={{ maxHeight: 350 }}>
          {error ? (
            <li className="px-4 py-6 text-center text-sm text-destructive">
              {error}
            </li>
          ) : users.length === 0 && !loading ? (
            <li className="px-4 py-6 text-center text-sm text-muted-foreground">
              No users yet.
            </li>
          ) : (
            users.map((u) => (
              <li key={u.id} className="flex items-center gap-3 px-4 py-3">
                <UserAvatar user={u} size="sm" />
                <div>
                  <p className="text-sm font-medium">{u.username}</p>
                  {u.full_name && (
                    <p className="text-xs text-muted-foreground">{u.full_name}</p>
                  )}
                </div>
              </li>
            ))
          )}
          {loading && (
            <li className="px-4 py-3 text-center text-xs text-muted-foreground">Loading…</li>
          )}
          {!loading && hasMore && (
            <li className="px-4 py-3 text-center">
              <button
                className="text-xs underline text-muted-foreground hover:text-foreground"
                onClick={loadMore}
              >
                Load more
              </button>
            </li>
          )}
        </ul>
      </div>
    </div>
  );
}

export default function ProfilePage() {
  const { username } = useParams<{ username: string }>();
  const { user: me } = useAuth();

  const {
    profile,
    isLoading,
    error,
    followers,
    following,
    profileSessions,
    isLoadingSessions,
    follow,
    unfollow,
    update,
  } = useProfile(username!);

  // Combined loading state for all profile data
  const isProfileLoading = isLoading || isLoadingSessions;

  const [editing, setEditing] = useState(false);
  const [bio, setBio] = useState('');
  const [fullName, setFullName] = useState('');
  const [showFollowers, setShowFollowers] = useState(false);
  const [showFollowing, setShowFollowing] = useState(false);
  const avatarInputRef = useRef<HTMLInputElement>(null);

  const isMe = me?.username === username;
  const isFollowing = followers.some((f) => f.id === me?.id);

  const handleFollow = () => {
    if (!profile) return;
    isFollowing ? unfollow.mutate(profile.id) : follow.mutate(profile.id);
  };

  const handleEditOpen = () => {
    setBio(profile?.bio ?? '');
    setFullName(profile?.full_name ?? '');
    setEditing(true);
  };

  const handleUpdate = async (e: React.FormEvent) => {
    e.preventDefault();
    await update.mutateAsync({ bio, full_name: fullName });
    setEditing(false);
  };

  // Avatar upload stub for TASK-069. Now with file type/size validation and error state.
  const [avatarError, setAvatarError] = useState<string | null>(null);
  const handleAvatarChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setAvatarError(null);
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setAvatarError('Please select an image file.');
      return;
    }
    if (file.size > 2 * 1024 * 1024) {
      setAvatarError('File size must be under 2MB.');
      return;
    }
    // Avatar upload will be implemented in TASK-069
  };

  if (isProfileLoading) {
    return (
      <PageShell>
        <p className="text-muted-foreground text-sm">Loading profile…</p>
      </PageShell>
    );
  }

  if (error) {
    return (
      <PageShell>
        <div className="text-center space-y-1 py-8">
          <p className="text-sm text-muted-foreground">Could not load profile.</p>
          <ErrorMessage error={error} />
        </div>
      </PageShell>
    );
  }

  if (!profile) {
    return (
      <PageShell>
        <p className="text-muted-foreground text-sm">User not found.</p>
      </PageShell>
    );
  }

  return (
    <PageShell>
      <div className="max-w-2xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-start gap-4">
          {/* Avatar — clickable upload trigger for own profile */}
          <div className="relative shrink-0">
            <UserAvatar user={profile} size="lg" />
            {isMe && (
              <>
                <button
                  onClick={() => avatarInputRef.current?.click()}
                  className="absolute inset-0 flex items-center justify-center rounded-full bg-black/40 opacity-0 hover:opacity-100 transition-opacity"
                  title="Change avatar"
                >
                  <span className="text-white text-xs font-medium">Edit</span>
                </button>
                <input
                  ref={avatarInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={handleAvatarChange}
                />
                {avatarError && (
                  <div className="absolute left-0 right-0 -bottom-6 text-xs text-destructive text-center">
                    {avatarError}
                  </div>
                )}
              </>
            )}
          </div>

          <div className="flex-1 space-y-1">
            <h1 className="text-xl font-bold">{profile.username}</h1>
            {profile.full_name && (
              <p className="text-muted-foreground text-sm">{profile.full_name}</p>
            )}
            {profile.bio && <p className="text-sm">{profile.bio}</p>}
            <div className="flex gap-4 text-sm text-muted-foreground pt-1">
              <button
                onClick={() => setShowFollowers(true)}
                className="hover:text-foreground transition-colors"
              >
                <strong className="text-foreground">{followers.length}</strong> followers
              </button>
              <button
                onClick={() => setShowFollowing(true)}
                className="hover:text-foreground transition-colors"
              >
                <strong className="text-foreground">{following.length}</strong> following
              </button>
            </div>
          </div>

          {isMe ? (
            <button
              onClick={editing ? () => setEditing(false) : handleEditOpen}
              className="rounded-md border px-3 py-1.5 text-sm font-medium"
            >
              {editing ? 'Cancel' : 'Edit profile'}
            </button>
          ) : (
            <button
              onClick={handleFollow}
              className={`rounded-md px-3 py-1.5 text-sm font-medium ${
                isFollowing ? 'border hover:bg-accent' : 'bg-foreground text-background'
              }`}
            >
              {isFollowing ? 'Unfollow' : 'Follow'}
            </button>
          )}
        </div>

        {/* Edit form */}
        {editing && (
          <form onSubmit={handleUpdate} className="space-y-3 rounded-xl border p-4">
            <div className="space-y-1.5">
              <label className="text-sm font-medium">Full name</label>
              <input
                type="text"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                className="w-full rounded-md border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-sm font-medium">Bio</label>
              <textarea
                value={bio}
                onChange={(e) => setBio(e.target.value)}
                rows={3}
                className="w-full rounded-md border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring resize-none"
              />
            </div>
            <button
              type="submit"
              disabled={update.isPending}
              className="rounded-md bg-foreground text-background px-4 py-2 text-sm font-medium disabled:opacity-60"
            >
              {update.isPending ? 'Saving…' : 'Save'}
            </button>
          </form>
        )}

        {/* Sessions */}
        <div className="space-y-4">
          <h2 className="text-lg font-semibold">
            {isMe ? 'Your sessions' : `${profile.username}'s sessions`}
          </h2>
          {isLoadingSessions ? (
            <p className="text-muted-foreground text-sm">Loading sessions…</p>
          ) : profileSessions.length === 0 ? (
            <p className="text-muted-foreground text-sm">No sessions yet.</p>
          ) : (
            <div className="grid gap-4 sm:grid-cols-2">
              {profileSessions.map((s) => (
                <SessionCard
                  key={s.id}
                  id={s.id}
                  date={new Date(s.created_at).toLocaleDateString()}
                  duration={Math.round(s.duration_seconds / 60)}
                  attentionPercentage={Math.round(s.attention_score)}
                  productivityScore={Math.round(s.attention_score)}
                  distractionCount={Math.round(s.total_attention_lost)}
                  subject={s.notes ?? 'Study Session'}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Followers modal */}
      {showFollowers && profile && (
        <UserListModal
          title="Followers"
          userId={profile.id}
          fetchUsers={getFollowers}
          onClose={() => setShowFollowers(false)}
        />
      )}

      {/* Following modal */}
      {showFollowing && profile && (
        <UserListModal
          title="Following"
          userId={profile.id}
          fetchUsers={getFollowing}
          onClose={() => setShowFollowing(false)}
        />
      )}
    </PageShell>
  );
}
