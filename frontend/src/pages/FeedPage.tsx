import { useQuery } from "@tanstack/react-query"
import { getFeed } from "@/api/feed"
import SessionCard from "@/components/shared/SessionCard"
import PageShell from "@/components/layout/PageShell"

export default function FeedPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["feed"],
    queryFn: () => getFeed(),
  })

  if (isLoading) return <p className="text-white p-8">Loading...</p>

  return (
    <PageShell>
      <div className="max-w-2xl mx-auto py-8 space-y-4">
        {data?.length === 0 && (
          <p className="text-slate-400 text-sm">
            No activity yet. Follow people to see their sessions here.
          </p>
        )}
        {data?.map(session => <SessionCard key={session.id} session={session} />)}
      </div>
    </PageShell>
  )
}

