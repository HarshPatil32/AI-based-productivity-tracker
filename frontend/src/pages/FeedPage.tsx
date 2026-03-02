import PageShell from '../components/layout/PageShell';
import SessionCard from '../components/shared/SessionCard';
import { useFeed } from '../hooks/useFeed';

export default function FeedPage() {
  const { data, isLoading, isFetchingNextPage, hasNextPage, fetchNextPage } = useFeed();

  const items = data?.pages.flatMap((p) => p.items) ?? [];

  return (
    <PageShell>
      <div className="max-w-2xl mx-auto space-y-6">
        <h1 className="text-2xl font-bold">Feed</h1>

        {isLoading && <p className="text-muted-foreground text-sm">Loading feed…</p>}

        {items.length === 0 && !isLoading && (
          <p className="text-muted-foreground text-sm">
            No activity yet. Follow people to see their sessions here.
          </p>
        )}

        <div className="space-y-4">
          {items.map((item) => (
            <div key={item.id} className="space-y-2">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <span className="font-medium text-foreground">
                  {item.author.username}
                </span>
                completed a session
              </div>
              <SessionCard session={item.session} />
            </div>
          ))}
        </div>

        {hasNextPage && (
          <button
            onClick={() => fetchNextPage()}
            disabled={isFetchingNextPage}
            className="w-full rounded-md border py-2 text-sm font-medium hover:bg-accent disabled:opacity-60"
          >
            {isFetchingNextPage ? 'Loading…' : 'Load more'}
          </button>
        )}
      </div>
    </PageShell>
  );
}
