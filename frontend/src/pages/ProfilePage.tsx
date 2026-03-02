import { useState } from 'react';
import { useParams } from 'react-router-dom';
import PageShell from '../components/layout/PageShell';
import UserAvatar from '../components/shared/UserAvatar';
import StatBadge from '../components/shared/StatBadge';
import SessionCard from '../components/shared/SessionCard';
import { useProfile } from '../hooks/useProfile';
import { useAuth } from '../hooks/useAuth';
import { useSessions } from '../hooks/useSessions';

export default function ProfilePage() {
  const { username } = useParams<{ username: string }>();
  const { user: me } = useAuth();
  const { profile, isLoading, followers, following, follow, unfollow, update } =
    useProfile(username!);
  const { sessions } = useSessions();
  const [editing, setEditing] = useState(false);
  const [bio, setBio] = useState('');

  const isMe = me?.username === username;
  const isFollowing = followers.some((f) => f.id === me?.id);

  const handleFollow = () => {
    if (!profile) return;
    isFollowing
      ? unfollow.mutate(profile.id)
      : follow.mutate(profile.id);
  };

  const handleUpdate = async (e: React.FormEvent) => {
    e.preventDefault();
    await update.mutateAsync({ bio });
    setEditing(false);
  };

  if (isLoading) {
    return (
      <PageShell>
        <p className="text-muted-foreground text-sm">Loading profile…</p>
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
          <UserAvatar user={profile} size="lg" />
          <div className="flex-1 space-y-1">
            <h1 className="text-xl font-bold">{profile.username}</h1>
            {profile.full_name && (
              <p className="text-muted-foreground text-sm">{profile.full_name}</p>
            )}
            {profile.bio && (
              <p className="text-sm">{profile.bio}</p>
            )}
            <div className="flex gap-4 text-sm text-muted-foreground pt-1">
              <span><strong className="text-foreground">{followers.length}</strong> followers</span>
              <span><strong className="text-foreground">{following.length}</strong> following</span>
            </div>
          </div>

          {isMe ? (
            <button
              onClick={() => { setEditing((v) => !v); setBio(profile.bio ?? ''); }}
              className="rounded-md border px-3 py-1.5 text-sm font-medium"
            >
              {editing ? 'Cancel' : 'Edit profile'}
            </button>
          ) : (
            <button
              onClick={handleFollow}
              className={`rounded-md px-3 py-1.5 text-sm font-medium ${
                isFollowing
                  ? 'border hover:bg-accent'
                  : 'bg-foreground text-background'
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
        {isMe && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold">Your sessions</h2>
            <div className="grid gap-4 sm:grid-cols-2">
              {sessions.map((s) => (
                <SessionCard key={s.id} session={s} />
              ))}
            </div>
            {sessions.length === 0 && (
              <p className="text-muted-foreground text-sm">No sessions yet.</p>
            )}
          </div>
        )}
      </div>
    </PageShell>
  );
}
