interface StatBadgeProps {
  label: string;
  value: string | number;
}

export default function StatBadge({ label, value }: StatBadgeProps) {
  return (
    <div className="flex flex-col items-center rounded-lg bg-muted px-3 py-1.5 min-w-[56px]">
      <span className="text-xs text-muted-foreground leading-none">{label}</span>
      <span className="text-sm font-semibold mt-0.5">{value}</span>
    </div>
  );
}
