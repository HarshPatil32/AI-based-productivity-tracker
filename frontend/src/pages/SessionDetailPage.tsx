import axios from 'axios';
import { useParams, useNavigate } from 'react-router-dom';
import PageShell from '../components/layout/PageShell';
import StatBadge from '../components/shared/StatBadge';
import ErrorMessage from '../components/shared/ErrorMessage';
import { useSession, useSessions } from '../hooks/useSessions';

export default function SessionDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { data: session, isLoading, error } = useSession(id);
  const { remove } = useSessions();

  const handleDelete = async () => {
    if (!id) return;
    if (!confirm('Delete this session?')) return;
    await remove.mutateAsync(id);
    navigate('/sessions');
  };

  if (!id) {
    return (
      <PageShell>
        <p className="text-muted-foreground text-sm">Session not found.</p>
      </PageShell>
    );
  }

  if (isLoading) {
    return (
      <PageShell>
        <p className="text-muted-foreground text-sm">Loading session…</p>
      </PageShell>
    );
  }

  if (error) {
    const is404 = axios.isAxiosError(error) && error.response?.status === 404;
    if (is404) {
      return (
        <PageShell>
          <p className="text-muted-foreground text-sm">Session not found.</p>
        </PageShell>
      );
    }
    return (
      <PageShell>
        <ErrorMessage error={error} />
      </PageShell>
    );
  }

  if (!session) {
    return (
      <PageShell>
        <p className="text-muted-foreground text-sm">Session not found.</p>
      </PageShell>
    );
  }

  const durationMin = session.duration_seconds
    ? Math.round(session.duration_seconds / 60)
    : null;

  return (
    <PageShell>
      <div className="max-w-2xl mx-auto space-y-6">
        <div>
          <h1 className="text-2xl font-bold">{session.notes ?? 'Study Session'}</h1>
          <p className="text-muted-foreground text-sm mt-1">
            {new Date(session.created_at).toLocaleDateString()}
          </p>
        </div>

        <div className="flex flex-wrap gap-3">
          {durationMin !== null && (
            <StatBadge label="Duration" value={`${durationMin}m`} />
          )}
          {session.attention_score !== undefined && (
            <StatBadge label="Attention" value={`${Math.round(session.attention_score)}%`} />
          )}
        </div>

        <div className="flex gap-3">
          <button
            onClick={handleDelete}
            disabled={remove.isPending}
            className="rounded-md border border-destructive text-destructive px-4 py-2 text-sm font-medium hover:bg-destructive/10 disabled:opacity-60"
          >
            Delete
          </button>
        </div>
      </div>
    </PageShell>
  );
}
