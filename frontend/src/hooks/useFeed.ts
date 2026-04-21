import { useInfiniteQuery } from '@tanstack/react-query';
import { getFeed, getGlobalFeed } from '../api/feed';

const PAGE_SIZE = 20;

export type FeedType = 'following' | 'global';

export function useFeed(type: FeedType = 'following') {
  return useInfiniteQuery({
    queryKey: ['feed', type],
    queryFn: ({ pageParam = 0 }) =>
      type === 'global'
        ? getGlobalFeed(PAGE_SIZE, pageParam as number)
        : getFeed(PAGE_SIZE, pageParam as number),
    initialPageParam: 0,
    getNextPageParam: (lastPage, allPages) =>
      lastPage.length === PAGE_SIZE ? allPages.length * PAGE_SIZE : undefined,
  });
}
