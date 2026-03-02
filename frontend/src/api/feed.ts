import apiClient from './client';
import type { FeedResponse } from '../types/feed';

export const getFeed = (params?: { page?: number; page_size?: number }) =>
  apiClient
    .get<FeedResponse>('/feed', { params })
    .then((r) => r.data);
