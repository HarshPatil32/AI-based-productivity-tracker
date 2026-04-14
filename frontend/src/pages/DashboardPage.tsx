import Navbar from '../components/layout/Navbar';
import ErrorMessage from '../components/shared/ErrorMessage';
import { useSessions } from '../hooks/useSessions';
import { Clock, Eye, Target, Flame, TrendingUp, Award, Calendar, Play } from 'lucide-react';

export default function DashboardPage() {
  const { sessions, isLoading, error } = useSessions();
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
          {[
            { icon: Clock, label: 'Total Hours' },
            { icon: Eye, label: 'Avg Attention' },
            { icon: Target, label: 'Productivity Score' },
            { icon: Flame, label: 'Current Streak' },
          ].map(({ icon: Icon, label }) => (
            <div key={label} className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <span className="text-sm font-medium text-gray-500">{label}</span>
                <Icon className="h-5 w-5 text-gray-400" />
              </div>
              <p className="text-2xl font-bold text-gray-400">—</p>
            </div>
          ))}
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Weekly Study Hours</h2>
            <div className="h-[250px] flex items-center justify-center text-sm text-gray-400">
              No data yet
            </div>
          </div>
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Attention Trend</h2>
            <div className="h-[250px] flex items-center justify-center text-sm text-gray-400">
              No data yet
            </div>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Recent Sessions */}
          <div className="lg:col-span-2">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold text-gray-900">Recent Sessions</h2>
              <button className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                <Play className="h-4 w-4 mr-2" />
                Start Session
              </button>
            </div>
            <div className="bg-white rounded-lg border border-gray-200 py-12 text-center">
              {error ? (
                <ErrorMessage error={error} />
              ) : isLoading ? (
                <p className="text-sm text-gray-400">Loading sessions…</p>
              ) : sessions.length === 0 ? (
                <p className="text-sm text-gray-400">No sessions yet. Start your first session!</p>
              ) : (
                <ul className="divide-y divide-gray-100 text-left">
                  {sessions.slice(0, 5).map((s) => (
                    <li key={s.id} className="px-6 py-3 flex items-center justify-between">
                      <span className="text-sm text-gray-700">
                        {new Date(s.started_at).toLocaleDateString()}
                      </span>
                      <span className="text-sm text-gray-500">
                        {Math.round(s.duration_seconds / 60)} min &middot; {s.attention_score}% attention
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
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
                  <span className="text-sm font-semibold text-gray-400">—</span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <TrendingUp className="h-5 w-5 text-gray-400 mr-3" />
                    <span className="text-sm text-gray-600">Avg Score</span>
                  </div>
                  <span className="text-sm font-semibold text-gray-400">—</span>
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
              <p className="text-sm text-blue-100 mb-4">25 hours of focused study</p>
              <div className="bg-white/20 rounded-full h-2 mb-2">
                <div className="bg-white rounded-full h-2" style={{ width: '0%' }}></div>
              </div>
              <p className="text-sm">0 / 25 hours (0%)</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
