import { Link } from 'react-router-dom';
import type { Session } from '../../types/session';
import StatBadge from './StatBadge';

interface SessionCardProps {
  session: Session;
}

export default function SessionCard({ session }: SessionCardProps) {
  const durationMin = session.duration_seconds
    ? Math.round(session.duration_seconds / 60)
    : null;

  return (
    <div className="rounded-xl border bg-card p-4 flex flex-col gap-3 hover:shadow-sm transition-shadow">
      <div className="flex items-start justify-between gap-2">
        <Link
          to={`/sessions/${session.id}`}
          className="font-semibold hover:underline truncate"
        >
          {session.title}
        </Link>
        <span
          className={`text-xs px-2 py-0.5 rounded-full font-medium shrink-0 ${
            session.status === 'active'
              ? 'bg-green-100 text-green-700'
              : session.status === 'paused'
              ? 'bg-yellow-100 text-yellow-700'
              : 'bg-muted text-muted-foreground'
          }`}
        >
          {session.status}
        </span>
      </div>

      {session.description && (
        <p className="text-sm text-muted-foreground line-clamp-2">
          {session.description}
        </p>
      )}

      <div className="flex flex-wrap gap-2">
        {durationMin !== null && (
          <StatBadge label="Duration" value={`${durationMin}m`} />
        )}
        {session.attention_score !== undefined && (
          <StatBadge
            label="Attention"
            value={`${Math.round(session.attention_score)}%`}
          />
        )}
        {session.focus_score !== undefined && (
          <StatBadge
            label="Focus"
            value={`${Math.round(session.focus_score)}%`}
          />
        )}
      </div>
    </div>
  );
}
