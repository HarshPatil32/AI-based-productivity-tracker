import { Calendar, Clock, Eye, TrendingUp } from 'lucide-react';

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
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-start justify-between">
        <div>
          <p className="font-semibold text-gray-900 text-lg">{subject}</p>
          <p className="flex items-center gap-1.5 text-sm text-gray-500 mt-1">
            <Calendar className="h-3.5 w-3.5" />
            {date}
          </p>
        </div>
        <span className="text-sm font-semibold bg-green-100 text-green-700 px-3 py-1 rounded-full whitespace-nowrap">
          {productivityScore} Score
        </span>
      </div>
      <div className="flex gap-8 mt-4">
        <div>
          <p className="flex items-center gap-1 text-xs text-gray-500 mb-1">
            <Clock className="h-3.5 w-3.5" />
            Duration
          </p>
          <p className="text-sm font-bold text-gray-900">{durationLabel}</p>
        </div>
        <div>
          <p className="flex items-center gap-1 text-xs text-gray-500 mb-1">
            <Eye className="h-3.5 w-3.5" />
            Attention
          </p>
          <p className="text-sm font-bold text-gray-900">{attentionPercentage}%</p>
        </div>
        <div>
          <p className="flex items-center gap-1 text-xs text-gray-500 mb-1">
            <TrendingUp className="h-3.5 w-3.5" />
            Distractions
          </p>
          <p className="text-sm font-bold text-gray-900">{distractionCount}</p>
        </div>
      </div>
    </div>
  );
}
