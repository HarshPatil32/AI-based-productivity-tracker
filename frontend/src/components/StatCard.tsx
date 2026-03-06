import type { LucideIcon } from 'lucide-react';

interface StatCardProps {
  icon: LucideIcon;
  label: string;
  value: string;
  change: string;
  changeType: 'positive' | 'negative' | 'neutral';
}

export function StatCard({ icon: Icon, label, value, change, changeType }: StatCardProps) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <span className="text-sm font-medium text-gray-500">{label}</span>
        <Icon className="h-5 w-5 text-gray-400" />
      </div>
      <p className="text-2xl font-bold text-gray-900">{value}</p>
      <p className={`text-xs mt-1 ${
        changeType === 'positive' ? 'text-green-600' :
        changeType === 'negative' ? 'text-red-600' :
        'text-gray-500'
      }`}>{change}</p>
    </div>
  );
}
