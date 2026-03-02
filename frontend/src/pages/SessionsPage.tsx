import { useState } from 'react';
import PageShell from '../components/layout/PageShell';
import SessionCard from '../components/shared/SessionCard';
import { useSessions } from '../hooks/useSessions';

export default function SessionsPage() {
  const { sessions, isLoading, create } = useSessions();
  const [title, setTitle] = useState('');
  const [showForm, setShowForm] = useState(false);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!title.trim()) return;
    await create.mutateAsync({ title: title.trim() });
    setTitle('');
    setShowForm(false);
  };

  return (
    <PageShell>
      <div className="max-w-3xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">Sessions</h1>
          <button
            onClick={() => setShowForm((v) => !v)}
            className="rounded-md bg-foreground text-background px-3 py-1.5 text-sm font-medium"
          >
            {showForm ? 'Cancel' : 'New session'}
          </button>
        </div>

        {showForm && (
          <form
            onSubmit={handleCreate}
            className="flex gap-2 rounded-xl border p-4"
          >
            <input
              type="text"
              placeholder="Session title…"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="flex-1 rounded-md border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            />
            <button
              type="submit"
              disabled={create.isPending}
              className="rounded-md bg-foreground text-background px-4 py-2 text-sm font-medium disabled:opacity-60"
            >
              Start
            </button>
          </form>
        )}

        {isLoading && <p className="text-muted-foreground text-sm">Loading…</p>}

        <div className="grid gap-4 sm:grid-cols-2">
          {sessions.map((session) => (
            <SessionCard key={session.id} session={session} />
          ))}
        </div>

        {sessions.length === 0 && !isLoading && (
          <p className="text-muted-foreground text-sm">No sessions yet. Start one!</p>
        )}
      </div>
    </PageShell>
  );
}
