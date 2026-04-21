import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider } from './store/authStore';
import ErrorBoundary from './components/ErrorBoundary';

import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import DashboardPage from './pages/DashboardPage';
import FeedPage from './pages/FeedPage';
import SessionsPage from './pages/SessionsPage';
import SessionDetailPage from './pages/SessionDetailPage';
import ProfilePage from './pages/ProfilePage';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 30_000,
    },
  },
});

function AppRoutes() {
  const { pathname } = useLocation();
  return (
    <Routes>
      {/* Public routes */}
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />

      {/* Protected routes (auth temporarily disabled) */}
      <Route path="/dashboard" element={<ErrorBoundary key={pathname}><DashboardPage /></ErrorBoundary>} />
      <Route path="/feed" element={<ErrorBoundary key={pathname}><FeedPage /></ErrorBoundary>} />
      <Route path="/sessions" element={<ErrorBoundary key={pathname}><SessionsPage /></ErrorBoundary>} />
      <Route path="/sessions/:id" element={<ErrorBoundary key={pathname}><SessionDetailPage /></ErrorBoundary>} />
      <Route path="/profile/:username" element={<ErrorBoundary key={pathname}><ProfilePage /></ErrorBoundary>} />

      {/* Default redirect */}
      <Route path="/" element={<Navigate to="/dashboard" replace />} />
      <Route path="*" element={<Navigate to="/dashboard" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <AppRoutes />
        </BrowserRouter>
      </QueryClientProvider>
    </AuthProvider>
  );
}
