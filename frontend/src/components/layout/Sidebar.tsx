import { NavLink } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';

const links = [
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/sessions', label: 'Sessions' },
  { to: '/feed', label: 'Feed' },
];

export default function Sidebar() {
  const { user } = useAuth();

  return (
    <aside className="hidden lg:flex flex-col w-56 shrink-0 border-r pt-14 h-screen sticky top-0">
      <nav className="flex flex-col gap-1 p-4">
        {links.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `px-3 py-2 rounded-md text-sm font-medium ${
                isActive
                  ? 'bg-accent text-accent-foreground'
                  : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
              }`
            }
          >
            {label}
          </NavLink>
        ))}
      </nav>

      {user && (
        <div className="mt-auto p-4 border-t">
          <NavLink
            to={`/profile/${user.username}`}
            className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
          >
            <div className="h-7 w-7 rounded-full bg-muted flex items-center justify-center text-xs font-medium uppercase">
              {user.username[0]}
            </div>
            <span className="truncate">{user.username}</span>
          </NavLink>
        </div>
      )}
    </aside>
  );
}
