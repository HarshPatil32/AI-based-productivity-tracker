import { useMemo, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell,
} from 'recharts';
import { Calendar, Download, Filter, Plus, Search, X } from 'lucide-react';
import PageShell from '../components/layout/PageShell';
import { SessionCard } from '../components/SessionCard';
import { useSessions, useSessionSummary } from '../hooks/useSessions';
import type { CreateSessionPayload } from '../types/session';

const EMPTY_FORM = { notes: '', started_at: '', ended_at: '' };

function SessionSkeleton() {
  return (
    <div className="space-y-4">
      {Array(4).fill(null).map((_, i) => (
        <div key={i} className="animate-pulse bg-gray-100 rounded-lg border border-gray-200 p-6 h-28" />
      ))}
    </div>
  );
}

interface LogSessionModalProps {
  onClose: () => void;
  onSubmit: (payload: CreateSessionPayload) => void;
  isPending: boolean;
}

function LogSessionModal({ onClose, onSubmit, isPending }: LogSessionModalProps) {
  const [form, setForm] = useState(EMPTY_FORM);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }));
    setError(null);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    const start = new Date(form.started_at);
    const end = new Date(form.ended_at);
    if (isNaN(start.getTime()) || isNaN(end.getTime())) {
      setError('Start and end times are required.');
      return;
    }
    if (end <= start) {
      setError('End time must be after start time.');
      return;
    }
    onSubmit({
      started_at: start.toISOString(),
      ended_at: end.toISOString(),
      duration_seconds: Math.round((end.getTime() - start.getTime()) / 1000),
      eyes_closed_time: 0,
      face_missing_time: 0,
      head_pose_off_time: 0,
      total_attention_lost: 0,
      notes: form.notes || undefined,
    });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div className="relative bg-white rounded-lg shadow-xl w-full max-w-md mx-4 p-6 space-y-5">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900">Log a session</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <X className="h-5 w-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-1">
              <label htmlFor="started_at" className="text-sm text-gray-600">Start time</label>
              <input
                id="started_at"
                name="started_at"
                type="datetime-local"
                value={form.started_at}
                onChange={handleChange}
                required
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <div className="space-y-1">
              <label htmlFor="ended_at" className="text-sm text-gray-600">End time</label>
              <input
                id="ended_at"
                name="ended_at"
                type="datetime-local"
                value={form.ended_at}
                onChange={handleChange}
                required
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>

          <div className="space-y-1">
            <label htmlFor="notes" className="text-sm text-gray-600">Subject / notes (optional)</label>
            <input
              id="notes"
              name="notes"
              type="text"
              placeholder="e.g. Calculus Chapter 4"
              value={form.notes}
              onChange={handleChange}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          {error && <p className="text-sm text-red-600">{error}</p>}

          <div className="flex justify-end gap-3 pt-1">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-lg border border-gray-300 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isPending}
              className="px-4 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isPending ? 'Saving...' : 'Save session'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

const PERF_COLORS = ['#22c55e', '#3b82f6', '#eab308', '#ef4444'];

export default function SessionsPage() {
  const { sessions, isLoading, isFetching, page, setPage, maxUnlockedPage, error, create } = useSessions();
  const { summary } = useSessionSummary();
  const [search, setSearch] = useState('');
  const [showModal, setShowModal] = useState(false);

  const filtered = useMemo(
    () => sessions.filter((s) =>
      (s.notes ?? '').toLowerCase().includes(search.toLowerCase())
    ),
    [sessions, search]
  );

  const subjectData = useMemo(() => {
    const map: Record<string, number> = {};
    sessions.forEach((s) => {
      const key = s.notes ?? 'Other';
      map[key] = (map[key] ?? 0) + s.duration_seconds / 3600;
    });
    return Object.entries(map)
      .map(([name, hours]) => ({ name, hours: Math.round(hours * 10) / 10 }))
      .sort((a, b) => b.hours - a.hours)
      .slice(0, 5);
  }, [sessions]);

  const performanceData = useMemo(() => {
    const counts = { Excellent: 0, Good: 0, Average: 0, 'Below Avg': 0 };
    sessions.forEach((s) => {
      if (s.attention_score >= 85) counts.Excellent++;
      else if (s.attention_score >= 70) counts.Good++;
      else if (s.attention_score >= 50) counts.Average++;
      else counts['Below Avg']++;
    });
    const total = sessions.length || 1;
    return [
      { name: 'Excellent', value: Math.round((counts.Excellent / total) * 100) },
      { name: 'Good', value: Math.round((counts.Good / total) * 100) },
      { name: 'Average', value: Math.round((counts.Average / total) * 100) },
      { name: 'Below Avg', value: Math.round((counts['Below Avg'] / total) * 100) },
    ].filter((d) => d.value > 0);
  }, [sessions]);

  const totalHours = summary ? (summary.total_study_seconds / 3600).toFixed(1) : '—';
  const avgScore = summary ? Math.round(summary.avg_attention_score) : '—';
  const pageNumbers = Array.from({ length: maxUnlockedPage }, (_, i) => i + 1);

  return (
    <PageShell>
      {showModal && (
        <LogSessionModal
          onClose={() => setShowModal(false)}
          onSubmit={(payload) => {
            create.mutate(payload, { onSuccess: () => setShowModal(false) });
          }}
          isPending={create.isPending}
        />
      )}
      <div className="max-w-7xl mx-auto space-y-8">

        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">My Sessions</h1>
            <p className="text-gray-600 mt-1">Track and analyze your study sessions</p>
          </div>
          <button
            onClick={() => setShowModal(true)}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Plus className="h-4 w-4" />
            Log session
          </button>
        </div>

        {/* Summary cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {[
            { label: 'Total Sessions', value: summary?.total_sessions ?? '—' },
            { label: 'Total Hours', value: summary ? `${totalHours}h` : '—' },
            { label: 'Avg Score', value: avgScore },
            { label: 'Avg Attention', value: summary ? `${avgScore}%` : '—' },
          ].map(({ label, value }) => (
            <div key={label} className="bg-white rounded-lg border border-gray-200 p-6">
              <p className="text-sm text-gray-600 mb-1">{label}</p>
              <p className="text-3xl font-bold text-gray-900">{value}</p>
            </div>
          ))}
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Study Hours by Subject</h2>
            {subjectData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={subjectData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis type="number" stroke="#6b7280" />
                  <YAxis type="category" dataKey="name" stroke="#6b7280" width={100} />
                  <Tooltip />
                  <Bar dataKey="hours" fill="#3b82f6" radius={[0, 8, 8, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[250px] flex items-center justify-center text-sm text-gray-400">
                No data yet
              </div>
            )}
          </div>

          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Performance Distribution</h2>
            {performanceData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={performanceData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }: { name?: string; percent?: number }) =>
                      `${name ?? ''} ${((percent ?? 0) * 100).toFixed(0)}%`
                    }
                    outerRadius={80}
                    dataKey="value"
                  >
                    {performanceData.map((_, i) => (
                      <Cell key={i} fill={PERF_COLORS[i % PERF_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[250px] flex items-center justify-center text-sm text-gray-400">
                No data yet
              </div>
            )}
          </div>
        </div>

        {/* Search / Filter bar */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search sessions..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full pl-10 pr-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none text-sm"
              />
            </div>
            <button className="flex items-center px-4 py-2 border border-gray-300 rounded-lg text-sm text-gray-700 hover:bg-gray-50 transition-colors">
              <Calendar className="h-4 w-4 mr-2" />
              Date Range
            </button>
            <button className="flex items-center px-4 py-2 border border-gray-300 rounded-lg text-sm text-gray-700 hover:bg-gray-50 transition-colors">
              <Filter className="h-4 w-4 mr-2" />
              Filter
            </button>
            <button className="flex items-center px-4 py-2 border border-gray-300 rounded-lg text-sm text-gray-700 hover:bg-gray-50 transition-colors">
              <Download className="h-4 w-4 mr-2" />
              Export
            </button>
          </div>
        </div>

        {/* Session list */}
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-4">All Sessions</h2>
          {isLoading ? (
            <SessionSkeleton />
          ) : error ? (
            <p className="text-sm text-red-600">Failed to load sessions. Please refresh and try again.</p>
          ) : filtered.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              {search ? 'No sessions match your search.' : 'No sessions yet. Use the tracker to record one.'}
            </p>
          ) : (
            <div className="space-y-4">
              {filtered.map((session) => (
                <SessionCard
                  key={session.id}
                  id={session.id}
                  date={new Date(session.created_at).toLocaleString('en-US', {
                    month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit',
                  })}
                  duration={Math.round(session.duration_seconds / 60)}
                  attentionPercentage={Math.round(session.attention_score)}
                  productivityScore={Math.round(session.attention_score)}
                  distractionCount={Math.round(session.total_attention_lost)}
                  subject={session.notes ?? 'Study Session'}
                />
              ))}
            </div>
          )}
        </div>

        {/* Numbered pagination */}
        {!isLoading && filtered.length > 0 && (
          <div className="flex justify-center">
            <nav className="flex items-center space-x-2">
              <button
                className="px-3 py-2 rounded-lg border border-gray-300 text-sm text-gray-700 hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1 || isFetching}
              >
                Previous
              </button>
              {pageNumbers.map((n) => (
                <button
                  key={n}
                  className={`px-3 py-2 rounded-lg text-sm ${
                    n === page
                      ? 'bg-blue-600 text-white'
                      : 'border border-gray-300 text-gray-700 hover:bg-gray-50'
                  } disabled:cursor-not-allowed`}
                  onClick={() => setPage(n)}
                  disabled={isFetching}
                >
                  {n}
                </button>
              ))}
              <button
                className="px-3 py-2 rounded-lg border border-gray-300 text-sm text-gray-700 hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed"
                onClick={() => setPage((p) => p + 1)}
                disabled={page === maxUnlockedPage || isFetching}
              >
                Next
              </button>
            </nav>
          </div>
        )}

      </div>
    </PageShell>
  );
}
