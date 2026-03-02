import PageShell from '../components/layout/PageShell';
import StatBadge from '../components/shared/StatBadge';
import { useAuth } from '../hooks/useAuth';
import { useSessions } from '../hooks/useSessions';

export default function DashboardPage() {
  const { user } = useAuth();
  const { sessions, isLoading } = useSessions();

  const completed = sessions.filter((s) => s.status === 'completed');
  const active = sessions.find((s) => s.status === 'active');

  const avgAttention =
    completed.length > 0
      ? Math.round(
          completed.reduce((acc, s) => acc + (s.attention_score ?? 0), 0) /
            completed.length
        )
      : null;

  return (
    <PageShell>
      <div className="max-w-3xl mx-auto space-y-8">
        <div>
          <h1 className="text-2xl font-bold">
            Welcome back{user ? `, ${user.username}` : ''}
          </h1>
          <p className="text-muted-foreground text-sm mt-1">Here's your focus summary.</p>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <StatBadge label="Total Sessions" value={sessions.length} />
          <StatBadge label="Completed" value={completed.length} />
          <StatBadge label="Avg Attention" value={avgAttention !== null ? `${avgAttention}%` : '—'} />
          <StatBadge label="Active Now" value={active ? 'Yes' : 'No'} />
        </div>

        {isLoading && (
          <p className="text-muted-foreground text-sm">Loading sessions…</p>
        )}
      </div>
    </PageShell>
  );
}
