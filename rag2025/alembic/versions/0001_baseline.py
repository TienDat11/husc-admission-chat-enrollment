"""Baseline schema: the 23 PostgreSQL tables from plan section PL-4.

Statements are executed verbatim in order (foreign-key safe). The audit_events
append-only guard (REVOKE + triggers raising AUDIT_APPEND_ONLY) is part of this
revision, not a comment.
"""

from alembic import op

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None

DDL_STATEMENTS = [
    "CREATE TABLE source_documents ( document_id text PRIMARY KEY, sha256 bytea NOT NULL, kind text NOT NULL CHECK (kind IN ('pdf','xlsx','docx','url','faq')), storage_uri text NOT NULL, original_name text NOT NULL, page_count int, admission_year int, status text NOT NULL DEFAULT 'uploaded' CHECK (status IN ('uploaded','parsing','parsed','failed','archived')), uploaded_by text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(), UNIQUE (sha256, kind))",
    'CREATE INDEX ix_source_documents_status_year ON source_documents (status, admission_year)',
    "CREATE TABLE parse_runs ( parse_run_id uuid PRIMARY KEY, document_id text NOT NULL REFERENCES source_documents, parser text NOT NULL, parser_version text NOT NULL, options jsonb NOT NULL DEFAULT '{}', page_routes jsonb NOT NULL DEFAULT '{}', status text NOT NULL CHECK (status IN ('running','succeeded','failed','cancelled')), error_code text, attempt int NOT NULL DEFAULT 1, created_at timestamptz NOT NULL DEFAULT now(), finished_at timestamptz)",
    'CREATE INDEX ix_parse_runs_doc ON parse_runs (document_id, created_at DESC)',
    "CREATE TABLE document_blocks ( parse_run_id uuid NOT NULL REFERENCES parse_runs ON DELETE CASCADE, block_id text NOT NULL, document_id text NOT NULL REFERENCES source_documents, page int NOT NULL CHECK (page >= 1), ordinal int NOT NULL CHECK (ordinal >= 0), block_type text NOT NULL CHECK (block_type IN ('text','table','heading','table_row')), text text NOT NULL, bbox jsonb, ocr_confidence numeric, table_tsv text, warnings jsonb NOT NULL DEFAULT '[]', version int NOT NULL DEFAULT 1, created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY (parse_run_id, block_id))",
    'CREATE INDEX ix_blocks_doc_page ON document_blocks (document_id, page, ordinal)',
    "CREATE TABLE canonical_records ( record_id text NOT NULL, version int NOT NULL CHECK (version >= 1), coordinate_key text NOT NULL, record_type text NOT NULL, scope text NOT NULL, subject text NOT NULL, admission_year int NOT NULL, admission_round text NOT NULL DEFAULT 'all', methods_hash text NOT NULL DEFAULT 'all', status text NOT NULL CHECK (status IN ('draft','review','approved','published','archived')), archive_reason text CHECK (archive_reason IN ('superseded','withdrawn','archived')), authority_level text NOT NULL, effective_from date NOT NULL, effective_to date, text text NOT NULL, structured_payload jsonb NOT NULL DEFAULT '{}', major_codes jsonb NOT NULL DEFAULT '[]', admission_methods jsonb NOT NULL DEFAULT '[]', source_document_id text REFERENCES source_documents, source_url text, source_page int, supersedes text, supersedes_version int, authored_by text NOT NULL, reviewed_by text, approved_by text, published_by text, published_at timestamptz, created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY (record_id, version), CHECK (record_id = coordinate_key), CHECK (approved_by IS NULL OR approved_by <> authored_by), CHECK (status <> 'published' OR source_document_id IS NOT NULL), CHECK ((status = 'archived') = (archive_reason IS NOT NULL)), CHECK (effective_to IS NULL OR effective_from <= effective_to))",
    "CREATE UNIQUE INDEX uq_current_coordinate ON canonical_records (coordinate_key) WHERE status = 'published'",
    "CREATE UNIQUE INDEX uq_one_open_draft ON canonical_records (record_id) WHERE status IN ('draft','review','approved')",
    'CREATE INDEX ix_records_status_year ON canonical_records (status, admission_year)',
    'CREATE INDEX ix_records_type_year ON canonical_records (record_type, admission_year)',
    'CREATE INDEX ix_records_major_codes ON canonical_records USING gin (major_codes)',
    "CREATE TABLE record_evidence ( evidence_id uuid PRIMARY KEY, record_id text NOT NULL, record_version int NOT NULL, document_id text NOT NULL REFERENCES source_documents, parse_run_id uuid REFERENCES parse_runs, page int NOT NULL CHECK (page >= 1), source_span int4range NOT NULL, block_ids jsonb NOT NULL DEFAULT '[]', excerpt text NOT NULL, FOREIGN KEY (record_id, record_version) REFERENCES canonical_records ON DELETE CASCADE)",
    'CREATE INDEX ix_evidence_record ON record_evidence (record_id, record_version)',
    "CREATE TABLE chunk_profiles ( key text NOT NULL, version int NOT NULL, record_types jsonb NOT NULL, strategy text NOT NULL CHECK (strategy IN ('legal_boundary','major_block','faq','table_row')), min_tokens int NOT NULL DEFAULT 50, target_tokens int NOT NULL DEFAULT 350, hard_max_tokens int NOT NULL DEFAULT 512, tokenizer text NOT NULL DEFAULT 'tiktoken:cl100k_base' CHECK (tokenizer = 'tiktoken:cl100k_base'), status text NOT NULL CHECK (status IN ('draft','active','retired')), definition_hash text NOT NULL, created_by text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY (key, version), CHECK (0 < min_tokens AND min_tokens < target_tokens AND target_tokens <= hard_max_tokens))",
    "CREATE UNIQUE INDEX uq_active_profile ON chunk_profiles (key) WHERE status = 'active'",
    "CREATE TABLE chunk_previews ( preview_id uuid PRIMARY KEY, parse_run_id uuid REFERENCES parse_runs, profile_key text NOT NULL, profile_version int NOT NULL, corpus_stamp jsonb NOT NULL, summary jsonb NOT NULL, status text NOT NULL CHECK (status IN ('ready','stale','failed')), release_id text, created_by text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), FOREIGN KEY (profile_key, profile_version) REFERENCES chunk_profiles)",
    'CREATE INDEX ix_previews_run ON chunk_previews (parse_run_id, created_at)',
    "CREATE TABLE chunk_preview_items ( preview_id uuid NOT NULL REFERENCES chunk_previews ON DELETE CASCADE, chunk_id text NOT NULL, record_id text NOT NULL, record_version int NOT NULL, parent_unit_id text, ordinal int NOT NULL, text text NOT NULL, token_count int NOT NULL CHECK (token_count BETWEEN 1 AND 512), metadata jsonb NOT NULL DEFAULT '{}', content_hash text NOT NULL, embedding bytea, embedding_model text, PRIMARY KEY (preview_id, chunk_id))",
    'CREATE INDEX ix_preview_items_record ON chunk_preview_items (record_id, record_version)',
    'CREATE TABLE ingestion_jobs ( job_id uuid PRIMARY KEY, kind text NOT NULL CHECK (kind IN (\'parse\',\'preview\',\'publish\',\'index\',\'retrieval_check\',\'reocr\')), payload jsonb NOT NULL, state text NOT NULL, priority int NOT NULL DEFAULT 100, attempt int NOT NULL DEFAULT 0, max_attempts int NOT NULL DEFAULT 3, progress jsonb NOT NULL DEFAULT \'{"done":0,"total":0,"stage":""}\', lease_owner text, lease_expires_at timestamptz, heartbeat_at timestamptz, available_at timestamptz NOT NULL DEFAULT now(), idempotency_key text NOT NULL, last_error jsonb, created_by text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(), UNIQUE (kind, idempotency_key))',
    "CREATE INDEX ix_jobs_claimable ON ingestion_jobs (priority, available_at, created_at) WHERE state IN ('queued','failed_retryable')",
    'CREATE INDEX ix_jobs_lease ON ingestion_jobs (lease_expires_at) WHERE lease_owner IS NOT NULL',
    "CREATE TABLE releases ( release_id text PRIMARY KEY, parent_release_id text REFERENCES releases, corpus_stamp jsonb NOT NULL, lancedb_table text NOT NULL, lancedb_version bigint, graphml_uri text, manifest jsonb NOT NULL, golden_report jsonb, status text NOT NULL CHECK (status IN ('building','gating','gate_failed','ready','active','superseded','rolled_back')), created_by text NOT NULL, activated_by text, created_at timestamptz NOT NULL DEFAULT now(), activated_at timestamptz)",
    "CREATE UNIQUE INDEX uq_one_active_release ON releases ((true)) WHERE status = 'active'",
    "CREATE TABLE reindex_outbox ( event_id uuid PRIMARY KEY, record_id text NOT NULL, record_version int NOT NULL, event_type text NOT NULL CHECK (event_type IN ('upsert','tombstone','supersede','revoke_supersession')), payload jsonb NOT NULL, release_id text REFERENCES releases, available_at timestamptz NOT NULL DEFAULT now(), processed_at timestamptz)",
    'CREATE INDEX ix_outbox_pending ON reindex_outbox (available_at) WHERE processed_at IS NULL',
    'CREATE TABLE audit_events ( event_id uuid PRIMARY KEY, seq bigint GENERATED ALWAYS AS IDENTITY UNIQUE, aggregate_type text NOT NULL, aggregate_id text NOT NULL, actor_id text NOT NULL, role text NOT NULL, action text NOT NULL, before jsonb, after jsonb, reason text, before_version int, after_version int, release_id text, request_id text, created_at timestamptz NOT NULL DEFAULT now(), prev_hash bytea, entry_hash bytea NOT NULL)',
    'CREATE INDEX ix_audit_aggregate ON audit_events (aggregate_type, aggregate_id, created_at)',
    "CREATE TABLE supersession_candidates ( candidate_id uuid PRIMARY KEY, document_id text NOT NULL REFERENCES source_documents, new_record_id text, new_record_version int, old_record_id text NOT NULL, old_record_version int NOT NULL, relation text NOT NULL CHECK (relation IN ('REPLACES','AMENDS','CORRECTS','REPEALS','EXPIRES','SPLITS','MERGES')), tier text NOT NULL CHECK (tier IN ('explicit_ref','coordinate_key','semantic')), similarity_score numeric, affected_articles jsonb NOT NULL DEFAULT '[]', detected_phrase text, decision text CHECK (decision IN ('supersede','repeal','temporal_handoff','co_exist')), status text NOT NULL DEFAULT 'pending_review' CHECK (status IN ('pending_review','accepted','rejected')), override_reason text, decided_by text, decided_at timestamptz, created_at timestamptz NOT NULL DEFAULT now(), FOREIGN KEY (old_record_id, old_record_version) REFERENCES canonical_records)",
    'CREATE INDEX ix_sc_doc_status ON supersession_candidates (document_id, status)',
    "CREATE TABLE conflict_entries ( conflict_id text PRIMARY KEY, coordinate_key text NOT NULL, record_ids jsonb NOT NULL, old_record_id text, new_record_id text, conflict_type text NOT NULL, field_diffs jsonb NOT NULL DEFAULT '[]', status text NOT NULL CHECK (status IN ('open','resolved','dismissed')), resolution text, override_reason text, resolved_by text, created_at timestamptz NOT NULL DEFAULT now(), resolved_at timestamptz)",
    'CREATE INDEX ix_conflicts_status ON conflict_entries (status, coordinate_key)',
    'CREATE TABLE supersession_retractions ( retraction_id uuid PRIMARY KEY, target_record_id text NOT NULL, superseded_version int NOT NULL, superseding_record_id text NOT NULL, superseding_version int NOT NULL, revoked_by text NOT NULL, revoked_at timestamptz NOT NULL, reason text NOT NULL CHECK (length(reason) >= 10), audit_event_id uuid REFERENCES audit_events)',
    'CREATE TABLE idempotency_keys ( key text PRIMARY KEY, scope text NOT NULL, request_hash text NOT NULL, status_code int NOT NULL, response_body jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), expires_at timestamptz NOT NULL)',
    'CREATE INDEX ix_idem_expires ON idempotency_keys (expires_at)',
    "CREATE TABLE admin_identities ( subject_id uuid PRIMARY KEY, actor_id text NOT NULL UNIQUE, role text NOT NULL CHECK (role IN ('viewer','editor','reviewer','publisher','reviewer_publisher','admin','ops')), password_hash text, header_token_hash bytea, disabled boolean NOT NULL DEFAULT false, failed_login_count int NOT NULL DEFAULT 0, locked_until timestamptz, last_login_at timestamptz, created_by text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(), CHECK (password_hash IS NOT NULL OR header_token_hash IS NOT NULL))",
    'CREATE TABLE admin_sessions ( session_id uuid PRIMARY KEY, subject_id uuid NOT NULL REFERENCES admin_identities ON DELETE CASCADE, actor_id text NOT NULL, role text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), expires_at timestamptz NOT NULL, revoked_at timestamptz, user_agent text)',
    'CREATE INDEX ix_sessions_subject ON admin_sessions (subject_id, revoked_at)',
    'CREATE INDEX ix_sessions_expires ON admin_sessions (expires_at)',
    "CREATE TABLE student_profiles ( student_session_id uuid PRIMARY KEY, raw_score numeric(5,2) CHECK (raw_score IS NULL OR raw_score BETWEEN 0 AND 30), method text CHECK (method IS NULL OR method IN ('thi_thpt','hoc_ba','dgnl','xet_tuyen_thang')), to_hop text CHECK (to_hop IS NULL OR to_hop ~ '^[A-Z][0-9]{2}$'), uu_tien_kv text CHECK (uu_tien_kv IS NULL OR uu_tien_kv IN ('KV1','KV2NT','KV2','KV3')), uu_tien_dt text CHECK (uu_tien_dt IS NULL OR uu_tien_dt IN ('UT1','UT2','NONE')), target_majors jsonb NOT NULL DEFAULT '[]', admission_year int, created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(), expires_at timestamptz NOT NULL DEFAULT now() + interval '180 days')",
    'CREATE INDEX ix_student_profiles_expires ON student_profiles (expires_at)',
    'CREATE TABLE chat_feedback ( feedback_id uuid PRIMARY KEY, trace_id text NOT NULL, helpful boolean NOT NULL, reason text, answered_at timestamptz NOT NULL, linked_item boolean NOT NULL DEFAULT false, release_id text, created_at timestamptz NOT NULL DEFAULT now())',
    'CREATE INDEX ix_feedback_trace ON chat_feedback (trace_id)',
    'CREATE INDEX ix_feedback_release ON chat_feedback (release_id, helpful)',
    'CREATE TABLE recommender_snapshots ( release_id text PRIMARY KEY REFERENCES releases, data_year int NOT NULL, diem_chuan_history jsonb NOT NULL, majors jsonb NOT NULL, to_hop jsonb NOT NULL, chi_tieu jsonb NOT NULL, methods jsonb NOT NULL, tuition jsonb NOT NULL, priority_rules jsonb NOT NULL, materialized_at timestamptz NOT NULL)',
    "CREATE TABLE golden_queries ( query_id text PRIMARY KEY, text text NOT NULL, expected_coordinate_keys jsonb NOT NULL, expected_pages jsonb NOT NULL DEFAULT '[]', query_embedding bytea, embedding_model text, expected_doctype text, record_type text NOT NULL, major_codes jsonb NOT NULL DEFAULT '[]', admission_year int, kind text NOT NULL CHECK (kind IN ('in_scope','out_of_scope','historical','supersession')), supersession_row smallint CHECK (supersession_row IS NULL OR supersession_row BETWEEN 1 AND 10), origin text NOT NULL CHECK (origin IN ('seed','promoted')), source_trace_id text, created_by text NOT NULL, created_at timestamptz NOT NULL DEFAULT now())",
    'CREATE INDEX ix_golden_type_year ON golden_queries (record_type, admission_year)',
    'CREATE INDEX ix_golden_majors ON golden_queries USING gin (major_codes)',
    'CREATE TABLE unanswered_questions ( trace_id text PRIMARY KEY, question text NOT NULL, normalized_hash text NOT NULL, occurrence_count int NOT NULL DEFAULT 1, asked_at timestamptz NOT NULL, route text NOT NULL, status_code text NOT NULL, resolved_at timestamptz, resolved_by_record_id text, out_of_scope boolean NOT NULL DEFAULT false, sla_due_at timestamptz NOT NULL)',
    'CREATE INDEX ix_unanswered_open ON unanswered_questions (sla_due_at) WHERE resolved_at IS NULL AND out_of_scope = false',
    'CREATE INDEX ix_unanswered_hash ON unanswered_questions (normalized_hash)',
]


AUDIT_GUARD_STATEMENTS = [
    """CREATE FUNCTION audit_events_reject_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'AUDIT_APPEND_ONLY'; END; $$""",
    "CREATE TRIGGER audit_events_no_update_delete BEFORE UPDATE OR DELETE ON audit_events FOR EACH ROW EXECUTE FUNCTION audit_events_reject_mutation()",
    "CREATE TRIGGER audit_events_no_truncate BEFORE TRUNCATE ON audit_events FOR EACH STATEMENT EXECUTE FUNCTION audit_events_reject_mutation()",
    """DO $$ BEGIN IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_app') THEN
REVOKE UPDATE, DELETE, TRUNCATE ON audit_events FROM rag_app; END IF; END $$""",
]

TABLES_DROP_ORDER = [
    "unanswered_questions",
    "golden_queries",
    "recommender_snapshots",
    "chat_feedback",
    "student_profiles",
    "admin_sessions",
    "admin_identities",
    "idempotency_keys",
    "supersession_retractions",
    "conflict_entries",
    "supersession_candidates",
    "audit_events",
    "reindex_outbox",
    "releases",
    "ingestion_jobs",
    "chunk_preview_items",
    "chunk_previews",
    "chunk_profiles",
    "record_evidence",
    "canonical_records",
    "document_blocks",
    "parse_runs",
    "source_documents",
]

def upgrade() -> None:
    conn = op.get_bind()
    for statement in DDL_STATEMENTS:
        conn.exec_driver_sql(statement)
    for statement in AUDIT_GUARD_STATEMENTS:
        conn.exec_driver_sql(statement)


def downgrade() -> None:
    conn = op.get_bind()
    for table in TABLES_DROP_ORDER:
        conn.exec_driver_sql(f"DROP TABLE IF EXISTS {table}")
    conn.exec_driver_sql("DROP FUNCTION IF EXISTS audit_events_reject_mutation()")
