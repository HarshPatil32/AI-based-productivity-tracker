import type { ReactNode } from 'react';
import Navbar from './Navbar';
import Sidebar from './Sidebar';

interface PageShellProps {
  children: ReactNode;
  /** Set to false to hide the sidebar (e.g. auth pages) */
  sidebar?: boolean;
}

export default function PageShell({ children, sidebar = true }: PageShellProps) {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <div className="flex flex-1">
        {sidebar && <Sidebar />}
        <main className="flex-1 overflow-y-auto p-6">{children}</main>
      </div>
    </div>
  );
}
