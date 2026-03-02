import { useInfiniteQuery } from '@tanstack/react-query';
import { getFeed } from '../api/feed';

const PAGE_SIZE = 20;

export function useFeed() {
  return useInfiniteQuery({
    queryKey: ['feed'],
    queryFn: ({ pageParam = 1 }) =>
      getFeed({ page: pageParam as number, page_size: PAGE_SIZE }),
    initialPageParam: 1,
    getNextPageParam: (lastPage) =>
      lastPage.has_next ? lastPage.page + 1 : undefined,
  });
}
