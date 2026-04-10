import { Link, NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';

export default function Navbar() {
  const { user, isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = async () => {
    await logout.mutateAsync();
    navigate('/login');
  };

  return (
    <nav className="fixed top-0 inset-x-0 z-50 h-14 border-b bg-background/80 backdrop-blur-sm flex items-center px-6 gap-6">
      <Link to="/" className="font-bold text-lg tracking-tight">
        FocusTrack
      </Link>

      {isAuthenticated && (
        <>
          <div className="flex items-center gap-4 flex-1">
            <NavLink
              to="/dashboard"
              className={({ isActive }) =>
                `text-sm font-medium ${isActive ? 'text-foreground' : 'text-muted-foreground hover:text-foreground'}`
              }
            >
              Dashboard
            </NavLink>
            <NavLink
              to="/sessions"
              className={({ isActive }) =>
                `text-sm font-medium ${isActive ? 'text-foreground' : 'text-muted-foreground hover:text-foreground'}`
              }
            >
              Sessions
            </NavLink>
            <NavLink
              to="/feed"
              className={({ isActive }) =>
                `text-sm font-medium ${isActive ? 'text-foreground' : 'text-muted-foreground hover:text-foreground'}`
              }
            >
              Feed
            </NavLink>
          </div>

          <div className="flex items-center gap-3 ml-auto">
            <NavLink
              to={`/profile/${user?.username}`}
              className="text-sm font-medium text-muted-foreground hover:text-foreground"
            >
              {user?.username}
            </NavLink>
            <button
              onClick={handleLogout}
              className="text-sm text-muted-foreground hover:text-foreground"
            >
              Logout
            </button>
          </div>
        </>
      )}

      {!isAuthenticated && (
        <div className="flex gap-3 ml-auto">
          <Link
            to="/login"
            className="text-sm font-medium text-muted-foreground hover:text-foreground"
          >
            Login
          </Link>
          <Link
            to="/register"
            className="text-sm font-medium bg-foreground text-background px-3 py-1.5 rounded-md"
          >
            Register
          </Link>
        </div>
      )}
    </nav>
  );
}
