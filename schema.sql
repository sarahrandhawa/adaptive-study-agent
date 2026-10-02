create table users (
    id uuid default gen_random_uuid() primary key,
    display_name text,
    created_at timestamptz default now(),
    email text
);

create table user_identities (
    user_id uuid references users(id) on delete restrict,
    provider text,
    subject text,
    unique(provider, subject)
);

create table courses(
    id uuid default gen_random_uuid() primary key,
    user_id uuid references users(id) on delete restrict,
    name text,
    created_at timestamptz default now(),
    archived_at timestamptz
);

create table documents(
    id uuid default gen_random_uuid() primary key,
    course_id uuid references courses(id) on delete restrict,
    user_id uuid references users(id) on delete restrict,
    filename text,
    content_hash text,
    extracted_text text,
    char_count int,
    chunk_count int,
    embedding_model text,
    status text,
    replaces_document_id uuid references documents(id) on delete restrict,
    created_at timestamptz default now(),
    deleted_at timestamptz
);

create table chunks(
    id uuid default gen_random_uuid() primary key,
    user_id uuid references users(id) on delete restrict,
    course_id uuid references courses(id) on delete restrict,
    document_id uuid references documents(id) on delete restrict,
    chunk_index int,
    text text,
    char_start int,
    char_end int, 
    created_at timestamptz default now(),
    deleted_at timestamptz
);

create table topics(
    id uuid default gen_random_uuid() primary key,
    user_id uuid references users(id) on delete restrict,
    course_id uuid references courses(id) on delete restrict,
    name text,
    normalized_key text,
    created_by text,
    created_at timestamptz default now(),
    unique(course_id, normalized_key)
);

create table questions(
    id uuid default gen_random_uuid() primary key,
    user_id uuid references users(id) on delete restrict,
    course_id uuid references courses(id) on delete restrict,
    question_type text,
    prompt text,
    reference_answer text,
    key_points jsonb,
    concept_label text,
    difficulty int,
    generation jsonb,
    status text,
    created_at timestamptz default now()
);

create table question_topics(
    user_id uuid references users(id) on delete restrict,
    question_id uuid references questions(id) on delete restrict,
    topic_id uuid references topics(id) on delete restrict,
    is_primary boolean,
    unique(question_id, topic_id)
);

create table question_sources(
    user_id uuid references users(id) on delete restrict,
    question_id uuid references questions(id) on delete restrict,
    chunk_id uuid references chunks(id) on delete restrict,
    rank int,
    unique(question_id, chunk_id)
);

create table attempts(
    id uuid default gen_random_uuid() primary key,
    user_id uuid references users(id) on delete restrict,
    question_id uuid references questions(id) on delete restrict,
    answer_text text,
    verdict text,
    score real,
    missed_points jsonb,
    feedback text,
    explanation text,
    identified_gap text,
    grader_model text,
    grader_version text,
    created_at timestamptz default now()
);

create table question_state (
    user_id uuid references users(id) on delete restrict,
    question_id uuid references questions(id) on delete restrict,
    mastery real,
    attempt_count int,
    correct_streak int,
    last_score real,
    last_attempt_at timestamptz,
    next_due_at timestamptz,
    updated_at timestamptz default now(),
    primary key (user_id, question_id)
);

create table usage_daily (
    user_id uuid references users(id) on delete restrict,
    day date,
    calls int,
    primary key (user_id, day)
);

alter table users enable row level security;
alter table user_identities enable row level security;
alter table courses enable row level security;
alter table documents enable row level security;
alter table chunks enable row level security;
alter table topics enable row level security;
alter table questions enable row level security;
alter table question_topics enable row level security;
alter table question_sources enable row level security;
alter table attempts enable row level security;
alter table question_state enable row level security;
alter table usage_daily enable row level security;