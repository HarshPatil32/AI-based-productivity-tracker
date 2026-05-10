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
-- SCHEMA AMENDMENTS
-- These ALTER TABLE statements must appear before any views that reference the
-- added columns, so they are grouped here rather than inside feature sections.
-- =============================================================================

-- likes_count was added after initial schema; kept as ALTER for idempotency
-- when applied to an existing database.
ALTER TABLE public.study_sessions
    ADD COLUMN IF NOT EXISTS likes_count INTEGER NOT NULL DEFAULT 0;


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


-- =============================================================================
-- ACHIEVEMENTS AND BADGES
-- =============================================================================

-- Achievement definitions (static catalogue, managed by admins / migrations)
--
-- metric_type drives which stat is compared against threshold_value when the
-- system evaluates whether a user has earned an achievement:
--   'sessions_count'       — total number of completed study sessions
--   'study_time_seconds'   — cumulative focused/session time in seconds
--   'streak_days'          — longest consecutive-day streak
--   'focus_score_percent'  — single-session focus score (0–100)
CREATE TABLE IF NOT EXISTS public.achievements (
    id               UUID    PRIMARY KEY DEFAULT uuid_generate_v4(),
    slug             TEXT    NOT NULL UNIQUE,
    name             TEXT    NOT NULL,
    description      TEXT,
    icon_url         TEXT,
    threshold_value  NUMERIC NOT NULL,
    metric_type      TEXT    NOT NULL
                         CHECK (metric_type IN (
                             'sessions_count',
                             'study_time_seconds',
                             'streak_days',
                             'focus_score_percent'
                         )),
    -- Soft-disable without losing unlock history
    is_active        BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Records which users have unlocked which achievements and when.
-- session_id is populated for session-scoped metrics (e.g. focus_score_percent)
-- to provide an audit trail; NULL for aggregate metrics.
CREATE TABLE IF NOT EXISTS public.achievement_unlocks (
    id             UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id        UUID        NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    achievement_id UUID        NOT NULL REFERENCES public.achievements(id) ON DELETE CASCADE,
    -- The specific session that triggered the unlock (nullable for non-session metrics)
    session_id     UUID        REFERENCES public.study_sessions(id) ON DELETE SET NULL,
    unlocked_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, achievement_id)
);

CREATE INDEX IF NOT EXISTS idx_achievement_unlocks_user_id
    ON public.achievement_unlocks (user_id);

CREATE INDEX IF NOT EXISTS idx_achievement_unlocks_achievement_id
    ON public.achievement_unlocks (achievement_id);

CREATE INDEX IF NOT EXISTS idx_achievements_metric_type
    ON public.achievements (metric_type);

-- Ensure session_id column exists (handles databases created from older schema versions)
ALTER TABLE public.achievement_unlocks
    ADD COLUMN IF NOT EXISTS session_id UUID REFERENCES public.study_sessions(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_achievement_unlocks_session_id
    ON public.achievement_unlocks (session_id)
    WHERE session_id IS NOT NULL;


-- =============================================================================
-- SEED: INITIAL ACHIEVEMENTS
-- =============================================================================

INSERT INTO public.achievements (slug, name, description, threshold_value, metric_type)
VALUES
    (
        'first-session',
        'First Session',
        'Complete your very first study session.',
        1,
        'sessions_count'
    ),
    (
        '10-hours-studied',
        '10 Hours Studied',
        'Accumulate 10 hours of total study time.',
        36000,
        'study_time_seconds'
    ),
    (
        '7-day-streak',
        '7-Day Streak',
        'Study for 7 consecutive days.',
        7,
        'streak_days'
    ),
    (
        '90-focus-master',
        '90% Focus Master',
        'Achieve a focus score of 90% or higher in a single session.',
        90,
        'focus_score_percent'
    ),
    (
        '100-sessions',
        '100 Sessions',
        'Complete 100 study sessions.',
        100,
        'sessions_count'
    )
ON CONFLICT (slug) DO NOTHING;


-- ---------------------------------------------------------------------------
-- recalculate_all_achievement_unlocks
--
-- Safety utility: re-evaluates and backfills achievement_unlocks for all
-- users based on current aggregate stats. Run after seeding new achievements
-- or after manual data changes to ensure unlock records are consistent.
--
-- Usage:  SELECT public.recalculate_all_achievement_unlocks();
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.recalculate_all_achievement_unlocks()
RETURNS VOID LANGUAGE plpgsql AS $$
DECLARE
    v_achievement RECORD;
    v_user        RECORD;
    v_stat        NUMERIC;
BEGIN
    FOR v_achievement IN
        SELECT id, metric_type, threshold_value
        FROM public.achievements
        WHERE is_active = TRUE
    LOOP
        FOR v_user IN SELECT id FROM public.profiles LOOP

            -- Compute the relevant stat for this user / metric combination
            IF v_achievement.metric_type = 'sessions_count' THEN
                SELECT COUNT(*)::NUMERIC INTO v_stat
                FROM public.study_sessions
                WHERE user_id = v_user.id;

            ELSIF v_achievement.metric_type = 'study_time_seconds' THEN
                SELECT COALESCE(SUM(session_duration), 0)::NUMERIC INTO v_stat
                FROM public.study_sessions
                WHERE user_id = v_user.id;

            ELSIF v_achievement.metric_type = 'streak_days' THEN
                -- Streak calculation is handled by the application layer;
                -- this function cannot recompute it from raw data alone.
                -- Skip streak achievements during bulk recalculation.
                CONTINUE;

            ELSIF v_achievement.metric_type = 'focus_score_percent' THEN
                SELECT COALESCE(MAX(focus_score), 0)::NUMERIC INTO v_stat
                FROM public.study_sessions
                WHERE user_id = v_user.id;
            END IF;

            -- Insert unlock if threshold is met; skip if already recorded
            IF v_stat >= v_achievement.threshold_value THEN
                INSERT INTO public.achievement_unlocks (user_id, achievement_id)
                VALUES (v_user.id, v_achievement.id)
                ON CONFLICT (user_id, achievement_id) DO NOTHING;
            END IF;

        END LOOP;
    END LOOP;
END;
$$;
