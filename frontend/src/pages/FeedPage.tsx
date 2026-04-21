import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import Navbar from '../components/layout/Navbar';
import { TrendingUp, Award, Users } from 'lucide-react';
import { useFeed, type FeedType } from '../hooks/useFeed';
import { getSuggestedUsers } from '../api/users';
import SessionCard from '../components/shared/SessionCard';
import UserAvatar from '../components/shared/UserAvatar';

export default function FeedPage() {
  const [tab, setTab] = useState<FeedType>('following');

  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading,
    isError,
  } = useFeed(tab);

  // Suggested users: fetch a small list excluding self and already-followed
  const { data: suggestedUsers } = useQuery({
    queryKey: ['users', 'suggested'],
    queryFn: () => getSuggestedUsers(5),
    staleTime: 5 * 60 * 1000,
  });

  // Deduplicate sessions by id
  const sessions = data
    ? Array.from(new Map(data.pages.flat().map((s) => [s.id, s])).values())
    : [];

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 pt-20">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Feed */}
          <div className="lg:col-span-2">
            <div className="mb-6">
              <h1 className="text-3xl font-bold text-gray-900 mb-2">Activity Feed</h1>
              <p className="text-gray-600">See what your network is studying</p>
            </div>

            {/* Filter Tabs */}
            <div className="bg-white rounded-lg border border-gray-200 mb-6">
              <div className="flex border-b border-gray-200">
                <button
                  className={`flex-1 px-4 py-3 text-sm font-medium ${
                    tab === 'following'
                      ? 'text-blue-600 border-b-2 border-blue-600'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                  onClick={() => setTab('following')}
                >
                  Following
                </button>
                <button
                  className={`flex-1 px-4 py-3 text-sm font-medium ${
                    tab === 'global'
                      ? 'text-blue-600 border-b-2 border-blue-600'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                  onClick={() => setTab('global')}
                >
                  Global
                </button>
              </div>
            </div>

            {/* Feed content */}
            {isLoading ? (
              <div className="bg-white rounded-lg border border-gray-200 py-16 text-center">
                <p className="text-gray-500 text-sm">Loading...</p>
              </div>
            ) : isError ? (
              <div className="bg-white rounded-lg border border-gray-200 py-16 text-center">
                <p className="text-red-500 text-sm">Failed to load feed. Please try again.</p>
              </div>
            ) : sessions.length === 0 ? (
              <div className="bg-white rounded-lg border border-gray-200 py-16 text-center">
                <p className="text-gray-500 text-sm">
                  {tab === 'following'
                    ? 'No posts yet. Follow someone to see their activity here.'
                    : 'No public sessions yet.'}
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {sessions.map((session) => (
                  <SessionCard key={session.id} session={session} />
                ))}

                {hasNextPage && (
                  <div className="text-center pt-2">
                    <button
                      onClick={() => fetchNextPage()}
                      disabled={isFetchingNextPage}
                      className="px-6 py-2 text-sm font-medium text-blue-600 border border-blue-300 rounded-lg hover:bg-blue-50 disabled:opacity-50"
                    >
                      {isFetchingNextPage ? 'Loading...' : 'Load More'}
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Suggested Users to Follow */}
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex items-center mb-4">
                <Users className="h-5 w-5 text-gray-700 mr-2" />
                <h3 className="text-lg font-semibold text-gray-900">Suggested Users</h3>
              </div>
              {suggestedUsers && suggestedUsers.length > 0 ? (
                <div className="space-y-3">
                  {suggestedUsers.slice(0, 5).map((user) => (
                    <div key={user.id} className="flex items-center gap-3">
                      <UserAvatar user={user} size="sm" />
                      <div className="min-w-0">
                        <p className="text-sm font-medium text-gray-900 truncate">
                          {user.full_name ?? user.username}
                        </p>
                        <p className="text-xs text-gray-500 truncate">@{user.username}</p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-gray-500">No suggestions available yet.</p>
              )}
            </div>

            {/* Trending Topics */}
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex items-center mb-4">
                <TrendingUp className="h-5 w-5 text-gray-700 mr-2" />
                <h3 className="text-lg font-semibold text-gray-900">Trending Topics</h3>
              </div>
              <p className="text-sm text-gray-500">No trending topics yet.</p>
            </div>

            {/* Weekly Leaderboard */}
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex items-center mb-4">
                <Award className="h-5 w-5 text-gray-700 mr-2" />
                <h3 className="text-lg font-semibold text-gray-900">Weekly Leaders</h3>
              </div>
              <p className="text-sm text-gray-500">No leaderboard data yet.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
