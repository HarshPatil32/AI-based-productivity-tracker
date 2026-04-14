import axios from 'axios';
import { cn } from '@/lib/utils';

interface Props {
  error: unknown;
  className?: string;
}

function extractMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    return error.response?.data?.detail ?? error.message ?? 'Something went wrong';
  }
  if (error instanceof Error) {
    return error.message;
  }
  return 'Something went wrong';
}

export default function ErrorMessage({ error, className }: Props) {
  if (!error) return null;
  return (
    <p className={cn('text-sm text-red-500', className)}>
      {extractMessage(error)}
    </p>
  );
}
