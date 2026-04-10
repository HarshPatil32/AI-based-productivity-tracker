import { useParams, useNavigate } from 'react-router-dom';
import PageShell from '../components/layout/PageShell';
import StatBadge from '../components/shared/StatBadge';
import { useSession, useSessions } from '../hooks/useSessions';

export default function SessionDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { data: session, isLoading } = useSession(Number(id));
  const { end, remove } = useSessions();

  const handleEnd = async () => {
    await end.mutateAsync(Number(id));
  };

  const handleDelete = async () => {
    if (!confirm('Delete this session?')) return;
    await remove.mutateAsync(Number(id));
    navigate('/sessions');
  };

  if (isLoading) {
    return (
      <PageShell>
        <p className="text-muted-foreground text-sm">Loading session…</p>
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
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold">{session.title}</h1>
            {session.description && (
              <p className="text-muted-foreground text-sm mt-1">{session.description}</p>
            )}
          </div>
          <span
            className={`text-xs px-2 py-0.5 rounded-full font-medium shrink-0 ${
              session.status === 'active'
                ? 'bg-green-100 text-green-700'
                : 'bg-muted text-muted-foreground'
            }`}
          >
            {session.status}
          </span>
        </div>

        <div className="flex flex-wrap gap-3">
          {durationMin !== null && (
            <StatBadge label="Duration" value={`${durationMin}m`} />
          )}
          {session.attention_score !== undefined && (
            <StatBadge label="Attention" value={`${Math.round(session.attention_score)}%`} />
          )}
          {session.focus_score !== undefined && (
            <StatBadge label="Focus" value={`${Math.round(session.focus_score)}%`} />
          )}
        </div>

        <div className="flex gap-3">
          {session.status === 'active' && (
            <button
              onClick={handleEnd}
              disabled={end.isPending}
              className="rounded-md bg-foreground text-background px-4 py-2 text-sm font-medium disabled:opacity-60"
            >
              {end.isPending ? 'Ending…' : 'End session'}
            </button>
          )}
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
