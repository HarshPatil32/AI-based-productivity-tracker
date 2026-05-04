-- =============================================================================
-- AI-based Productivity Tracker — Database Schema
-- Target: Supabase (PostgreSQL)
-- =============================================================================

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


-- =============================================================================
-- TABLES
-- =============================================================================

-- User profiles (public data, mirrors auth.users)
CREATE TABLE IF NOT EXISTS public.profiles (
    id               UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    username         TEXT NOT NULL UNIQUE,
    full_name        TEXT,
    avatar_url       TEXT,
    bio              TEXT,
    total_study_time INTEGER NOT NULL DEFAULT 0,  -- cumulative seconds
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Study sessions recorded by the attention tracker
CREATE TABLE IF NOT EXISTS public.study_sessions (
    id                 UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id            UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,

    -- Display / metadata
    title              TEXT NOT NULL,
    description        TEXT,
    is_public          BOOLEAN NOT NULL DEFAULT TRUE,

    -- Duration breakdown (seconds)
    session_duration   INTEGER NOT NULL DEFAULT 0,
    focused_time       INTEGER NOT NULL DEFAULT 0,
    distracted_time    INTEGER NOT NULL DEFAULT 0,
    eyes_closed_time   INTEGER NOT NULL DEFAULT 0,
    face_missing_time  INTEGER NOT NULL DEFAULT 0,
    head_pose_off_time INTEGER NOT NULL DEFAULT 0,
    attention_lost     INTEGER NOT NULL DEFAULT 0,

    -- Derived scores (0–100)
    focus_score        FLOAT NOT NULL DEFAULT 0,
    attention_score    FLOAT NOT NULL DEFAULT 0,

    -- Quality label: 'Excellent' | 'Good' | 'Fair' | 'Poor'
    quality            TEXT NOT NULL DEFAULT 'Fair',

    -- Timing
    session_date       DATE,
    session_start_time TIMESTAMPTZ,
    session_end_time   TIMESTAMPTZ,

    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Social follow graph
CREATE TABLE IF NOT EXISTS public.user_relationships (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    follower_id  UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    following_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (follower_id, following_id)
);

-- Per-user application settings
CREATE TABLE IF NOT EXISTS public.user_settings (
    user_id              UUID PRIMARY KEY REFERENCES public.profiles(id) ON DELETE CASCADE,

    -- Privacy
    profile_visibility   TEXT NOT NULL DEFAULT 'public'
                             CHECK (profile_visibility IN ('public', 'friends', 'private')),
    session_visibility   TEXT NOT NULL DEFAULT 'public'
                             CHECK (session_visibility IN ('public', 'friends', 'private')),
    show_study_time      BOOLEAN NOT NULL DEFAULT TRUE,
    show_focus_scores    BOOLEAN NOT NULL DEFAULT TRUE,

    -- Notifications
    email_notifications  BOOLEAN NOT NULL DEFAULT TRUE,
    email_on_like        BOOLEAN NOT NULL DEFAULT TRUE,
    email_on_comment     BOOLEAN NOT NULL DEFAULT TRUE,
    email_on_follow      BOOLEAN NOT NULL DEFAULT TRUE,

    -- Display preferences
    theme                TEXT NOT NULL DEFAULT 'light'
                             CHECK (theme IN ('light', 'dark', 'system')),
    language             TEXT NOT NULL DEFAULT 'en',
    timezone             TEXT NOT NULL DEFAULT 'UTC',

    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- =============================================================================
-- INDEXES
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_study_sessions_user_id
    ON public.study_sessions (user_id);

CREATE INDEX IF NOT EXISTS idx_study_sessions_created_at
    ON public.study_sessions (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_study_sessions_is_public
    ON public.study_sessions (is_public)
    WHERE is_public = TRUE;

CREATE INDEX IF NOT EXISTS idx_user_relationships_follower_id
    ON public.user_relationships (follower_id);

CREATE INDEX IF NOT EXISTS idx_user_relationships_following_id
    ON public.user_relationships (following_id);


-- =============================================================================
-- VIEWS
-- =============================================================================

-- ---------------------------------------------------------------------------
-- user_profile_summary
--
-- Enriches each profile row with aggregated session statistics and social
-- counts. Referenced by:
--   - GET  /api/v1/users/me
--   - PATCH /api/v1/users/me  (re-fetch after update)
--   - GET  /api/v1/users/{username}
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW public.user_profile_summary AS
SELECT
    p.id,
    p.username,
    p.full_name,
    p.avatar_url,
    p.bio,
    p.total_study_time,
    p.created_at,

    -- Session statistics
    COUNT(DISTINCT ss.id)::INTEGER           AS total_sessions,
    AVG(ss.focus_score)                      AS avg_focus_score,

    -- Social counts
    COUNT(DISTINCT followers.follower_id)::INTEGER  AS followers_count,
    COUNT(DISTINCT following.following_id)::INTEGER AS following_count

FROM public.profiles p
LEFT JOIN public.study_sessions   ss        ON ss.user_id       = p.id
LEFT JOIN public.user_relationships followers ON followers.following_id = p.id
LEFT JOIN public.user_relationships following ON following.follower_id  = p.id
GROUP BY
    p.id,
    p.username,
    p.full_name,
    p.avatar_url,
    p.bio,
    p.total_study_time,
    p.created_at;


-- ---------------------------------------------------------------------------
-- feed_sessions
--
-- Denormalized view of public study sessions joined with author profile
-- information. Used by the social feed endpoints:
--   - GET /api/v1/feed/
--   - GET /api/v1/feed/global
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW public.feed_sessions AS
SELECT
    ss.id,
    ss.user_id,

    -- Author profile fields
    p.username,
    p.full_name,
    p.avatar_url,

    -- Session content
    ss.title,
    ss.description,
    ss.session_duration,
    ss.focused_time,
    ss.focus_score,
    ss.attention_score,
    ss.quality,

    -- Timing
    TO_CHAR(ss.session_date, 'YYYY-MM-DD') AS session_date,
    ss.session_start_time,
    ss.session_end_time,

    -- Engagement counters
    ss.likes_count,
    0 AS comments_count,

    ss.created_at

FROM public.study_sessions ss
JOIN public.profiles p ON p.id = ss.user_id
WHERE ss.is_public = TRUE;


-- =============================================================================
-- SESSION LIKES
-- =============================================================================

-- Add likes_count to study_sessions if it does not already exist
ALTER TABLE public.study_sessions
    ADD COLUMN IF NOT EXISTS likes_count INTEGER NOT NULL DEFAULT 0;

-- One like per user per session
CREATE TABLE IF NOT EXISTS public.session_likes (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES public.study_sessions(id) ON DELETE CASCADE,
    user_id    UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_session_likes_session_id
    ON public.session_likes (session_id);

CREATE INDEX IF NOT EXISTS idx_session_likes_user_id
    ON public.session_likes (user_id);

-- Trigger: keep study_sessions.likes_count accurate on insert / delete.
-- Assumptions:
--   - All likes are inserted/deleted through the session_likes table.
--   - Direct modifications to study_sessions.likes_count are not expected.
--   - If likes are ever manually deleted from the table outside the API,
--     run recalculate_all_likes_counts() to restore accuracy.
CREATE OR REPLACE FUNCTION public.update_session_likes_count()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE public.study_sessions
        SET likes_count = likes_count + 1
        WHERE id = NEW.session_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE public.study_sessions
        SET likes_count = GREATEST(likes_count - 1, 0)
        WHERE id = OLD.session_id;
    END IF;
    RETURN NULL;
END;
$$;

DROP TRIGGER IF EXISTS trg_session_likes_count ON public.session_likes;
CREATE TRIGGER trg_session_likes_count
AFTER INSERT OR DELETE ON public.session_likes
FOR EACH ROW EXECUTE FUNCTION public.update_session_likes_count();

-- ---------------------------------------------------------------------------
-- recalculate_all_likes_counts
--
-- Safety utility: recomputes likes_count for every session directly from
-- the session_likes table. Run this after any manual data changes or to
-- recover from inconsistencies.
--
-- Usage:  SELECT public.recalculate_all_likes_counts();
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.recalculate_all_likes_counts()
RETURNS VOID LANGUAGE plpgsql AS $$
BEGIN
    UPDATE public.study_sessions ss
    SET likes_count = (
        SELECT COUNT(*)::INTEGER
        FROM public.session_likes sl
        WHERE sl.session_id = ss.id
    );
END;
$$;


-- =============================================================================
-- SESSION COMMENTS
-- =============================================================================

-- Deleting a parent comment cascades to all its replies (ON DELETE CASCADE on
-- parent_comment_id). This is intentional: replies are meaningless without
-- their parent. Max nesting depth of 1 is enforced by the API layer.
CREATE TABLE IF NOT EXISTS public.session_comments (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id        UUID NOT NULL REFERENCES public.study_sessions(id) ON DELETE CASCADE,
    user_id           UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    parent_comment_id UUID REFERENCES public.session_comments(id) ON DELETE CASCADE,
    content           TEXT NOT NULL CHECK (char_length(content) BETWEEN 1 AND 1000),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_session_comments_session_id
    ON public.session_comments (session_id);

CREATE INDEX IF NOT EXISTS idx_session_comments_parent_id
    ON public.session_comments (parent_comment_id)
    WHERE parent_comment_id IS NOT NULL;
