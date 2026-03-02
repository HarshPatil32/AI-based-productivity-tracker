import { api } from "./client"
import type { FeedSession } from "@/types/feed"

export const getFeed = (limit = 20, offset = 0) =>
  api.get<FeedSession[]>("/feed/", { params: { limit, offset } }).then(r => r.data)
