interface SessionCardProps {
  id: string;
  date: string;
  duration: number;
  attentionPercentage: number;
  productivityScore: number;
  distractionCount: number;
  subject: string;
}

export function SessionCard({ date, duration, attentionPercentage, productivityScore, distractionCount, subject }: SessionCardProps) {
  const hours = Math.floor(duration / 60);
  const mins = duration % 60;
  const durationLabel = hours > 0 ? `${hours}h ${mins}m` : `${mins}m`;

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="flex items-start justify-between">
        <div>
          <p className="font-semibold text-gray-900">{subject}</p>
          <p className="text-xs text-gray-500 mt-0.5">{date}</p>
        </div>
        <span className="text-sm font-medium text-gray-700">{durationLabel}</span>
      </div>
      <div className="flex gap-4 mt-3 text-sm text-gray-600">
        <span>Attention: <strong>{attentionPercentage}%</strong></span>
        <span>Score: <strong>{productivityScore}</strong></span>
        <span>Distractions: <strong>{distractionCount}</strong></span>
      </div>
    </div>
  );
}
