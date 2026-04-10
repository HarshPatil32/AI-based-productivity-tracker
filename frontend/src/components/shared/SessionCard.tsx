import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import type { FeedSession } from "@/types/feed"

const qualityColor: Record<string, string> = {
  Excellent: "bg-green-500",
  Good: "bg-blue-500",
  Fair: "bg-yellow-500",
  Poor: "bg-red-500",
}

export default function SessionCard({ session }: { session: FeedSession }) {
  const mins = Math.round(session.session_duration / 60)
  return (
    <Card className="bg-slate-900 border-slate-800 text-white">
      <CardHeader className="flex flex-row items-center gap-3">
        <div>
          <p className="font-semibold">{session.username}</p>
          <p className="text-xs text-slate-400">{session.session_date}</p>
        </div>
      </CardHeader>
      <CardContent className="space-y-2">
        <p className="font-medium">{session.title}</p>
        <div className="flex gap-2 flex-wrap">
          <Badge className={qualityColor[session.quality] ?? "bg-gray-500"}>{session.quality}</Badge>
          <Badge variant="outline">{mins} min</Badge>
          <Badge variant="outline">Focus: {session.focus_score}%</Badge>
        </div>
      </CardContent>
    </Card>
  )
}
