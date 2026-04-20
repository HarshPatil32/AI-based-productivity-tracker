import { useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { Clock, Eye, Target, Flame, TrendingUp, Award, Calendar, Play } from 'lucide-react';
import Navbar from '../components/layout/Navbar';
import { StatCard } from '../components/StatCard';
import { SessionCard } from '../components/SessionCard';
import ErrorMessage from '../components/shared/ErrorMessage';
import { useSessions, useSessionSummary } from '../hooks/useSessions';
import type { Session } from '../types/session';

const WEEKLY_GOAL_HOURS = 25;

function toLocalDateStr(d: Date): string {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

function StatSkeleton() {
  return <div className="animate-pulse bg-gray-100 rounded-lg border border-gray-200 p-6 h-28" />;
}

function SessionSkeleton() {
  return (
    <div className="space-y-4">
      {Array(3).fill(null).map((_, i) => (
        <div key={i} className="animate-pulse bg-gray-100 rounded-lg border border-gray-200 p-6 h-28" />
      ))}
    </div>
  );
}

export default function DashboardPage() {
  const navigate = useNavigate();
  const { sessions, isLoading: sessionsLoading, error: sessionsError } = useSessions();
  const { summary, isLoading: summaryLoading, error: summaryError } = useSessionSummary();

  // Derive last-7-days chart buckets from the first fetched page of sessions.
  // Both bucket key and session key use local time so near-midnight sessions
  // are attributed to the correct local day.
  const chartData = useMemo(() => {
    const today = new Date();
    const days = Array.from({ length: 7 }, (_, i) => {
      const d = new Date(today);
      d.setDate(today.getDate() - (6 - i));
      return {
        day: d.toLocaleDateString('en-US', { weekday: 'short' }),
        date: toLocalDateStr(d),
        hours: 0,
        attention: 0,
        count: 0,
      };
    });

    for (const s of sessions) {
      const bucket = days.find((d) => d.date === toLocalDateStr(new Date(s.started_at)));
      if (bucket) {
        bucket.hours += s.duration_seconds / 3600;
        bucket.attention += s.attention_score;
        bucket.count += 1;
      }
    }

    return days.map((d) => ({
      day: d.day,
      hours: parseFloat(d.hours.toFixed(1)),
      attention: d.count > 0 ? parseFloat((d.attention / d.count).toFixed(0)) : 0,
    }));
  }, [sessions]);

  const hasChartData = chartData.some((d) => d.hours > 0);

  // Hours this ISO week for the weekly goal progress bar
  const weeklyHours = useMemo(() => {
    const now = new Date();
    const startOfWeek = new Date(now);
    startOfWeek.setDate(now.getDate() - now.getDay());
    startOfWeek.setHours(0, 0, 0, 0);
    return sessions.reduce((acc, s) => {
      return new Date(s.started_at) >= startOfWeek ? acc + s.duration_seconds / 3600 : acc;
    }, 0);
  }, [sessions]);

  const weeklyPct = Math.min((weeklyHours / WEEKLY_GOAL_HOURS) * 100, 100);

  // Current streak — computed from the fetched sessions (first page, up to 20)
  const currentStreak = useMemo(() => {
    if (!sessions.length) return 0;
    const sessionDays = new Set(sessions.map((s) => toLocalDateStr(new Date(s.started_at))));
    const today = new Date();
    const todayStr = toLocalDateStr(today);
    const yesterday = new Date(today);
    yesterday.setDate(today.getDate() - 1);
    // If neither today nor yesterday has a session the streak is broken
    const hasToday = sessionDays.has(todayStr);
    const hasYesterday = sessionDays.has(toLocalDateStr(yesterday));
    if (!hasToday && !hasYesterday) return 0;
    const cursor = new Date(hasToday ? today : yesterday);
    let streak = 0;
    while (sessionDays.has(toLocalDateStr(cursor))) {
      streak++;
      cursor.setDate(cursor.getDate() - 1);
    }
    return streak;
  }, [sessions]);

  // Monthly stats — filter current-month sessions from the already-fetched page
  const { monthlyCount, monthlyAvgScore } = useMemo(() => {
    const now = new Date();
    const monthly = sessions.filter((s) => {
      const d = new Date(s.started_at);
      return d.getMonth() === now.getMonth() && d.getFullYear() === now.getFullYear();
    });
    const count = monthly.length;
    const avgScore =
      count > 0 ? monthly.reduce((acc, s) => acc + s.attention_score, 0) / count : 0;
    return { monthlyCount: count, monthlyAvgScore: avgScore };
  }, [sessions]);

  // Recent sessions (fix: ensure this is defined)
  const recentSessions: Session[] = sessions.slice(0, 5);

  // Summary-derived stat values — optional chaining guards against error states
  // where summaryLoading is false but summary is still undefined.
  const totalHours = summary ? ((summary.total_study_seconds ?? 0) / 3600).toFixed(1) + 'h' : '—';
  const avgAttention = summary ? (summary.avg_attention_score ?? 0).toFixed(0) + '%' : '—';
  const productivityScore = summary ? (summary.avg_focus_score ?? 0).toFixed(0) + '%' : '—';

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 pt-20">
        {/* Welcome Section */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600 mt-1">Your productivity overview</p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {summaryError && <div className="col-span-4"><ErrorMessage error={summaryError} /></div>}
          {summaryLoading ? (
            Array(4).fill(null).map((_, i) => <StatSkeleton key={i} />)
          ) : (
            <>
              <StatCard icon={Clock} label="Total Hours" value={totalHours} change="" changeType="neutral" />
              <StatCard icon={Eye} label="Avg Attention" value={avgAttention} change="" changeType="neutral" />
              <StatCard icon={Target} label="Productivity Score" value={productivityScore} change="" changeType="neutral" />
              <StatCard icon={Flame} label="Current Streak" value={currentStreak > 0 ? `${currentStreak}d` : '—'} change="" changeType="neutral" />
            </>
          )}
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Weekly Study Hours</h2>
            {hasChartData ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="day" />
                  <YAxis unit="h" />
                  <Tooltip formatter={(v) => [`${v ?? 0}h`, 'Hours']} />
                  <Bar dataKey="hours" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[250px] flex items-center justify-center text-sm text-gray-400">
                No data yet
              </div>
            )}
          </div>

          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Attention Trend</h2>
            {hasChartData ? (
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="day" />
                  <YAxis unit="%" domain={[0, 100]} />
                  <Tooltip formatter={(v) => [`${v ?? 0}%`, 'Attention']} />
                  <Line
                    type="monotone"
                    dataKey="attention"
                    stroke="#6366f1"
                    strokeWidth={2}
                    dot={{ r: 4 }}
                    activeDot={{ r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[250px] flex items-center justify-center text-sm text-gray-400">
                No data yet
              </div>
            )}
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Recent Sessions */}
          <div className="lg:col-span-2">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold text-gray-900">Recent Sessions</h2>
              <button
                onClick={() => navigate('/sessions')}
                className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                <Play className="h-4 w-4 mr-2" />
                Start Session
              </button>
            </div>

            {sessionsError && <ErrorMessage error={sessionsError} className="mb-4" />}
            {sessionsLoading ? (
              <SessionSkeleton />
            ) : recentSessions.length > 0 ? (
              <div className="space-y-4">
                {recentSessions.map((s: Session) => (
                  <SessionCard
                    key={s.id}
                    id={s.id}
                    date={new Date(s.started_at).toLocaleDateString()}
                    duration={Math.round(s.duration_seconds / 60)}
                    attentionPercentage={Math.round(s.attention_score)}
                    productivityScore={Math.round(s.focus_score)}
                    distractionCount={0}
                    subject={s.notes ?? 'Study Session'}
                  />
                ))}
              </div>
            ) : (
              <div className="bg-white rounded-lg border border-gray-200 py-12 text-center">
                <p className="text-sm text-gray-400">No sessions yet. Start your first session!</p>
              </div>
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Quick Stats */}
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">This Month</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <Calendar className="h-5 w-5 text-gray-400 mr-3" />
                    <span className="text-sm text-gray-600">Sessions</span>
                  </div>
                  <span className="text-sm font-semibold text-gray-900">
                    {sessionsLoading ? '—' : monthlyCount}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <TrendingUp className="h-5 w-5 text-gray-400 mr-3" />
                    <span className="text-sm text-gray-600">Avg Score</span>
                  </div>
                  <span className="text-sm font-semibold text-gray-900">
                    {sessionsLoading ? '—' : monthlyCount > 0 ? monthlyAvgScore.toFixed(0) + '%' : '—'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <Award className="h-5 w-5 text-gray-400 mr-3" />
                    <span className="text-sm text-gray-600">Rank</span>
                  </div>
                  <span className="text-sm font-semibold text-gray-400">—</span>
                </div>
              </div>
            </div>

            {/* Recent Achievements */}
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Achievements</h3>
              <p className="text-sm text-gray-400">No achievements yet. Keep studying!</p>
            </div>

            {/* Weekly Goal */}
            <div className="bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg p-6 text-white">
              <h3 className="text-lg font-semibold mb-2">Weekly Goal</h3>
              <p className="text-sm text-blue-100 mb-4">{WEEKLY_GOAL_HOURS} hours of focused study</p>
              <div className="bg-white/20 rounded-full h-2 mb-2">
                <div
                  className="bg-white rounded-full h-2 transition-all"
                  style={{ width: `${weeklyPct.toFixed(0)}%` }}
                />
              </div>
              <p className="text-sm">
                {weeklyHours.toFixed(1)} / {WEEKLY_GOAL_HOURS} hours ({weeklyPct.toFixed(0)}%)
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
