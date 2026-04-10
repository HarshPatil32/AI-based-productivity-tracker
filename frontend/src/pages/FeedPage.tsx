import Navbar from '../components/layout/Navbar';
import { TrendingUp, Users, Award, Search } from 'lucide-react';

export default function FeedPage() {
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
                <button className="flex-1 px-4 py-3 text-sm font-medium text-blue-600 border-b-2 border-blue-600">
                  Following
                </button>
                <button className="flex-1 px-4 py-3 text-sm font-medium text-gray-500 hover:text-gray-700">
                  Trending
                </button>
                <button className="flex-1 px-4 py-3 text-sm font-medium text-gray-500 hover:text-gray-700">
                  Global
                </button>
              </div>
            </div>

            {/* Empty state */}
            <div className="bg-white rounded-lg border border-gray-200 py-16 text-center">
              <p className="text-gray-500 text-sm">No posts yet. Follow someone to see their activity here.</p>
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Search */}
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search users, topics..."
                  className="w-full pl-10 pr-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                />
              </div>
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

            {/* Study Groups */}
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex items-center mb-4">
                <Users className="h-5 w-5 text-gray-700 mr-2" />
                <h3 className="text-lg font-semibold text-gray-900">Suggested Groups</h3>
              </div>
              <p className="text-sm text-gray-500">No suggested groups yet.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
