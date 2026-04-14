import PageShell from '../components/layout/PageShell';
import { SessionCard } from '../components/SessionCard';
import { useSessions } from '../hooks/useSessions';

export default function SessionsPage() {
  const { sessions, isLoading } = useSessions();

  return (
    <PageShell>
      <div className="max-w-3xl mx-auto space-y-6">
        <h1 className="text-2xl font-bold">Sessions</h1>

        {isLoading && <p className="text-muted-foreground text-sm">Loading…</p>}

        <div className="grid gap-4 sm:grid-cols-2">
          {sessions.map((session) => (
            <SessionCard
              key={session.id}
              id={session.id}
              date={new Date(session.created_at).toLocaleDateString()}
              duration={Math.round(session.duration_seconds / 60)}
              attentionPercentage={Math.round(session.attention_score)}
              productivityScore={Math.round(session.attention_score)}
              distractionCount={Math.round(session.total_attention_lost)}
              subject={session.notes ?? 'Study Session'}
            />
          ))}
        </div>

        {sessions.length === 0 && !isLoading && (
          <p className="text-muted-foreground text-sm">No sessions yet. Use the tracker to record one.</p>
        )}
      </div>
    </PageShell>
  );
}
