import type { User } from '../../types/auth';

interface UserAvatarProps {
  user: Pick<User, 'username' | 'avatar_url'>;
  size?: 'sm' | 'md' | 'lg';
}

const sizeMap = {
  sm: 'h-7 w-7 text-xs',
  md: 'h-9 w-9 text-sm',
  lg: 'h-12 w-12 text-base',
};

export default function UserAvatar({ user, size = 'md' }: UserAvatarProps) {
  const classes = `rounded-full bg-muted flex items-center justify-center font-semibold uppercase shrink-0 ${sizeMap[size]}`;

  if (user.avatar_url) {
    return (
      <img
        src={user.avatar_url}
        alt={user.username}
        className={`${classes} object-cover`}
      />
    );
  }

  return <div className={classes}>{user.username[0]}</div>;
}
