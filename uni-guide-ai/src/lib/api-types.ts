export type paths = {
    "/": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Root
         * @description API info
         */
        get: operations["root__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/audit/events": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Events
         * @description The most recent events, newest first, matching every supplied filter.
         *
         *     Newest first because the question being asked is almost always "what just
         *     changed"; the chain order is still visible in ``seq``, which descends down
         *     the page and never skips a number within a page.
         */
        get: operations["list_events_admin_audit_events_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/audit/events/{event_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Event
         * @description One event by its id, with the chain position it was stored at.
         */
        get: operations["get_event_admin_audit_events__event_id__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/audit/verify": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Verify Chain
         * @description Whether the ledger is intact, and where it first is not.
         *
         *     A 200 with ``ok: false`` rather than a 5xx: a broken chain is an answer, not
         *     a server failure, and the caller needs ``broken_index`` to investigate.
         */
        get: operations["verify_chain_admin_audit_verify_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/conflicts": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Conflicts
         * @description Refresh the queue from current data and return every item.
         *
         *     Detection runs on read so the queue reflects the corpus without a manual
         *     command (criterion 1); it never selects a winner, so refreshing is safe
         *     in front of a live officer mid-decision.
         */
        get: operations["list_conflicts_admin_conflicts_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/conflicts/{conflict_id}/resolve": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Resolve Conflict
         * @description Record the override decision for one conflict.
         *
         *     The reason must be non-blank *here* (422) — the first of the two
         *     enforcement points. The actor and role handed to the service are the
         *     identity the credential resolved to, so a body claiming to be somebody
         *     else has nothing to claim with.
         */
        post: operations["resolve_conflict_admin_conflicts__conflict_id__resolve_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/identities": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Identities
         * @description Every stored identity, credentials masked.
         */
        get: operations["list_identities_admin_identities_get"];
        put?: never;
        /**
         * Create Identity
         * @description Register a credential for an actor and record why.
         *
         *     ``before_role`` is ``None`` - the event says "this subject had no grants
         *     before", which is what makes a creation distinguishable in the ledger
         *     from a change down to ``viewer``.
         */
        post: operations["create_identity_admin_identities_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/identities/{subject_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        /**
         * Patch Identity
         * @description Change one identity's role, or disable it, and record the change.
         *
         *     Disabling is a role change like any other: the ledger sees
         *     ``after_role=None`` and can answer "when did this person lose access?".
         *     The entry is never deleted, because a removed row would take the target
         *     of the event with it.
         */
        patch: operations["patch_identity_admin_identities__subject_id__patch"];
        trace?: never;
    };
    "/admin/identity": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Identity
         * @description Who the presented credential is, and what it may do.
         *
         *     No action is required to read this, deliberately: the response exposes only
         *     the caller's *own* grants — never another role's — and a role granted
         *     nothing must still be able to discover that it was granted nothing. A
         *     required action here would make that case unanswerable (403 on the very
         *     call that reports the 403s).
         */
        get: operations["get_identity_admin_identity_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/index-jobs": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Index Jobs
         * @description Index jobs: pending/running/failed/active, plus builds saved but not serving.
         *
         *     ``serving`` is true only for an ``active`` job. A build that exists in the
         *     staging area is reported ``saved=true, serving=false`` — never as published.
         */
        get: operations["list_index_jobs_admin_index_jobs_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/login": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Login
         * @description Authenticate by actor_id + password and set the session cookie.
         *
         *     Returns 401 for a wrong password or no configured password; 423 (Locked)
         *     when the actor is in a lockout window.
         */
        post: operations["login_admin_login_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/logout": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Logout
         * @description Clear the session cookie. Idempotent — a request with no cookie is a no-op.
         */
        post: operations["logout_admin_logout_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/records": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Records
         * @description Every stored record's latest version, filtered by ngành / năm / trạng thái.
         *
         *     Filters combine with AND; an absent filter matches everything. Newest
         *     admission year first, then ``record_id`` — a total order, so the same store
         *     lists in the same order on every call.
         */
        get: operations["list_records_admin_records_get"];
        put?: never;
        /**
         * Create Record
         * @description Store a new canonical record as version 1 (draft).
         */
        post: operations["create_record_admin_records_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/records/{record_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Record
         * @description Current version of one record.
         */
        get: operations["get_record_admin_records__record_id__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/records/{record_id}/claim": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Claim Record
         * @description Attach the caller as author of an unclaimed draft (ticket T15).
         *
         *     The seed drafts carry ``authored_by = null`` (spec 02 §108), and a version
         *     with no recorded author cannot be approved — four-eyes refuses the
         *     approval rather than approve for an unknown author. This route is the fix
         *     that keeps the seed version: it appends a new version whose ``authored_by``
         *     is the caller, the seed version and its audit history stay as written.
         *
         *     Only legal on a draft that has no author yet: a non-draft is a 409
         *     (claiming would demote a version a decision has been made on) and an
         *     already-claimed draft is a 409 too — claim is not the way to re-author
         *     someone else's work, that is an edit.
         */
        post: operations["claim_record_admin_records__record_id__claim_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/records/{record_id}/edits": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Edit Record
         * @description Append a new version of a record.
         *
         *     The appended version returns to ``draft``: the previous version's review no
         *     longer describes this content. A published version is never rewritten.
         */
        post: operations["edit_record_admin_records__record_id__edits_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/records/{record_id}/review-diff": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Review Diff
         * @description Diff of the draft (or a given version) vs the published version.
         */
        get: operations["review_diff_admin_records__record_id__review_diff_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/records/{record_id}/review-preview": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Review Preview
         * @description The answer a student would receive if this draft were published.
         */
        post: operations["review_preview_admin_records__record_id__review_preview_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/records/{record_id}/transitions": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Transition Record
         * @description Move a record along the lifecycle state machine.
         *
         *     One route carries five grants, so the required action is resolved from the
         *     payload and checked before the store is touched: ``submit`` is an editor's,
         *     ``approve``/``reject`` a reviewer's, ``publish``/``supersede`` a
         *     publisher's. A lifecycle action the matrix does not map is refused — an
         *     action nobody has been granted is not one this route may perform.
         *
         *     An illegal move is refused with 409 and writes nothing; a reject without a
         *     reason is refused with 422; an approve by the version's author is refused
         *     with 403 (four-eyes). ``supersede`` archives a published record only when
         *     another published record already names it in ``supersedes``.
         *
         *     Idempotency (ticket 27): the state machine already makes a *second* publish
         *     impossible — ``_TRANSITIONS`` has no self-loop, so a replay is refused 409
         *     rather than applied twice. That is state-based protection, and it is not the
         *     same as a replay returning the first call's answer: an operator whose network
         *     dropped mid-publish would get a 409 on retry and no way to tell "my publish
         *     landed" from "someone else published". With an ``Idempotency-Key`` the retry
         *     returns the original response instead.
         */
        post: operations["transition_record_admin_records__record_id__transitions_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/records/{record_id}/versions": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Record Versions
         * @description Every stored version of one record, newest first.
         */
        get: operations["get_record_versions_admin_records__record_id__versions_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/releases": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Release Status
         * @description Which release is active, and which one is behind it.
         *
         *     Read-only: the pointer and the manifests, straight off disk. The previous
         *     release is the one a targetless ``POST /admin/rollback`` would move to.
         */
        get: operations["get_release_status_admin_releases_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/reload-table": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Reload Table
         * @description Drain → build new retriever → healthcheck → atomic swap → close old.
         *
         *     @spec(S13.4) atomic blue-green swap (M1+F1+F4 from plan v2.1).
         *
         *     The swap itself is the ``app._swap_runtime_services`` seam (``src/main.py``),
         *     which rebinds every place a query can read a retriever. Assigning
         *     ``state.retriever`` alone left every query path — /query, /v2, /v3 — on the
         *     old table while this route reported a successful reload.
         *
         *     A drain that does not finish inside ``drain_timeout_s`` is a refusal, not a
         *     warning: nothing is swapped, the old retriever is not closed, and the answer
         *     is 409. Swapping anyway would close a retriever a request is still reading,
         *     which is the AttributeError ``src/app/request_tracker.py`` documents. Only an
         *     explicit ``force_while_in_flight`` swaps past a timeout.
         */
        post: operations["reload_table_admin_reload_table_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/rollback": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Rollback Release
         * @description Roll canonical + retrieval back to the last good release, together.
         *
         *     Both levels move as one unit: the canonical pointer (step C), then the
         *     LanceDB tag and the running runtime (step D). If step D fails, the pre-call
         *     state is restored (step E) and the 5xx names both failures. The ROLLBACK
         *     audit event is emitted only after both levels have moved, so the ledger
         *     never records a rollback that did not happen.
         *
         *     The same ``_reload_lock`` a blue-green reload holds is taken here: a reload
         *     and a rollback both change what every query reads, so they must not overlap.
         *
         *     The same ``request_tracker.gate`` a blue-green reload holds is taken here
         *     too, for the same reason: step D rebinds the runtime every query path reads,
         *     so an in-flight query still holding the old retriever would be answered from
         *     a release the ledger just recorded as rolled back. Admission reopens once
         *     step D has completed.
         *
         *     A drain that times out is refused with 409 (FIX-01), exactly as the reload
         *     route refuses one: the old "log and proceed" policy answered a request from
         *     the release being rolled away from, and closed the table under it. The
         *     rollback that must happen anyway is ``force_while_in_flight=true`` — an
         *     explicit emergency decision by the caller, recorded in the response.
         *
         *     Idempotency (ticket 27): ``target_release_id`` is optional, and without it
         *     the target is resolved from the *current* pointer. A replayed implicit
         *     rollback therefore resolves a *different* target — R3→R2, then R2→R1 — and
         *     walks back two releases for what the caller believed was one action. No
         *     state comparison catches that (the two calls legitimately differ), so an
         *     ``Idempotency-Key`` is the only thing that can. With one, a replay returns
         *     the first call's stored response and does not run again.
         */
        post: operations["rollback_release_admin_rollback_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/source-documents": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Source Documents
         * @description Every registered source document, sorted by id.
         */
        get: operations["list_source_documents_admin_source_documents_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/source-documents/backfill": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Backfill Source Documents
         * @description Register every file-backed document the record store already cites.
         *
         *     Derived from the store, so a citation nobody thought to list is still
         *     registered; corpus pointers are counted and skipped by design. Safe to
         *     repeat — a second call reports ``already_registered`` and writes nothing.
         */
        post: operations["backfill_source_documents_admin_source_documents_backfill_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/source-documents/blind-citations": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Blind Citations
         * @description Record versions citing a document the registry does not hold.
         *
         *     Read-only and non-blocking: nothing here stops a publish (Ghi chú 2 — the
         *     enforce step is a later ticket, once this list is clean for real).
         */
        get: operations["list_blind_citations_admin_source_documents_blind_citations_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/source-documents/content": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Source Document Content
         * @description Serve the registered source file as raw bytes.
         *
         *     ``document_id`` is looked up verbatim, resolved through the registry
         *     (so a ``..`` escape is refused), and read back raw: a returned body is
         *     byte-identical to the file on disk. Missing entry / path-escape / absent
         *     file all answer 404; an unauthenticated caller answers 422 per the
         *     admin surface's established convention.
         */
        get: operations["get_source_document_content_admin_source_documents_content_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/source-documents/from-url": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Ingest From Url
         * @description Fetch a pasted notification URL and draft / refresh its record.
         *
         *     - the same URL with the same bytes again → ``unchanged``, nothing written;
         *     - changed bytes → a new version of the same record, back at ``draft``;
         *     - localhost / private / link-local hosts and non-http(s) schemes are 400
         *       before any socket is opened;
         *     - the page is streamed under a byte cap, and the fetch is retried with
         *       backoff on 5xx and transport errors, then 502.
         */
        post: operations["ingest_from_url_admin_source_documents_from_url_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/source-documents/register": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Register Source Document
         * @description Register a source document that already sits in the repo.
         *
         *     Registration by reference, not upload: the file must be at
         *     ``repo_root / document_id``. Its sha256 is taken now; a missing file
         *     registers as ``file_present=false`` (an observation, not an error — the
         *     PDFs are gitignored).
         */
        post: operations["register_source_document_admin_source_documents_register_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/source-documents/registered": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Check Registered
         * @description Answer "is this document id in the registry?" for one id.
         *
         *     The id is a query parameter, never a path segment: it carries slashes and
         *     Vietnamese, and the response echoes it back unchanged so a caller can see
         *     the verbatim round trip.
         */
        get: operations["check_registered_admin_source_documents_registered_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/source-documents/retire": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Retire Source Document
         * @description Stop vouching for a document; the entry stays as history.
         */
        post: operations["retire_source_document_admin_source_documents_retire_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/source-documents/update": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Update Source Document
         * @description Re-observe a registered document (new hash and/or new note).
         *
         *     This is how a file whose content changed after records cited it becomes
         *     visible: the response's ``sha256`` differs from the previous revision's,
         *     and the audit event names both revisions.
         */
        post: operations["update_source_document_admin_source_documents_update_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/source-documents/upload": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Upload Source Document
         * @description Upload a PDF/DOCX/XLSX, read it in a subprocess, and draft records.
         *
         *     - magic bytes decide the kind; the extension is not read at all, so
         *       a ``.exe`` renamed ``.pdf`` is 415;
         *     - the same bytes again are 409 (the registry's content-hash key);
         *     - 20 MB cap and 300-page cap are 400;
         *     - a decompression bomb is refused *in the parser subprocess* and the
         *       route answers 400 while ``/v2/query`` keeps serving — the
         *       isolation is the whole point of the ticket;
         *     - an XLSX drafts one record per data row (ticket T17).
         */
        post: operations["upload_source_document_admin_source_documents_upload_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/unanswered/": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Unanswered
         * @description Open unanswered questions, newest first, with the resolution markers.
         *
         *     By default only open items are returned (the ones the team still needs to
         *     review); pass ``resolved=true`` to see everything, including items T15
         *     already resolved.
         */
        get: operations["list_unanswered_admin_unanswered__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/admin/warmup": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Warmup
         * @description Pre-warm heavy services (embedding model, LLM connection).
         *     Call after deploy before serving production traffic.
         *     @spec(S10.B7)
         */
        post: operations["warmup_admin_warmup_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/meta": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Meta
         * @description Return banner-ready meta data: current year + freshness status.
         *
         *     Resolution order for freshness fields:
         *       1. app.state.retriever.metadata_snapshot() if present.
         *       2. rag2025/data/audit_pre_reingest.json fallback.
         *       3. None (FE renders with env year only).
         *
         *     Contract: `retriever.metadata_snapshot()` MUST be a pure in-memory read.
         *     If the retriever needs I/O to compute snapshot data, it should pre-cache
         *     those values during startup. Otherwise wrap the call in asyncio.to_thread
         *     to avoid blocking the event loop.
         */
        get: operations["get_meta_api_meta_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/debug/preview-chunks": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Preview Chunks
         * @description Preview chunking results without saving to disk
         *
         *     This endpoint is useful for UAT and testing chunking quality
         */
        post: operations["preview_chunks_debug_preview_chunks_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/health": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Health Check
         * @description Health check with LanceDB connection + warmup readiness contract.
         *
         *     New fields (Phase 1 of warmup-ux):
         *       - `ready: bool`  — True when all core subsystems are wired up.
         *       - `stage: 'warming'|'ready'|'failed'`  — set by the lifespan state machine.
         *       - `services: dict`  — per-subsystem boolean map for diagnostics.
         *       - `reranker_load_ok: bool`  — ticket 40: the reranker is exactly as
         *         configured (loaded or intentionally off); False only when the model
         *         is dead or was never initialized.
         *
         *     Legacy fields (`status`, `lancedb_connected`, `embedding_model`,
         *     `reranker_model`, `metrics`) are preserved for backward compat.
         */
        get: operations["health_check_health_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/metrics": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Metrics Endpoint
         * @description Lightweight internal metrics endpoint.
         */
        get: operations["metrics_endpoint_metrics_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/query": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Query
         * @description Main RAG endpoint with HYDE multi-variant enhancement
         *
         *     Flow:
         *     1. HYDE: user query → 3-5 query variants
         *     2. Embedding: encode ALL variants to vectors
         *     3. LanceDB retrieval: retrieve chunks for ALL variants
         *     4. Merge & Deduplicate: combine results from all variants
         *     5. LLM: generate answer from merged chunks
         */
        post: operations["query_query_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v2/compare": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Compare
         * @description Build a same-year comparison table for several majors.
         *
         *     The comparison contract lives in :class:`ComparisonResponse`; read its
         *     docstring for the three guarantees (same effective year, per-cell
         *     provenance, missing cells marked not inferred).
         */
        post: operations["compare_v2_compare_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v2/feedback": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Feedback
         * @description Student 👍/👎 feedback on a served answer.
         *
         *     Public endpoint: no authentication — it is the student talking to the
         *     system, not the office. Idempotent by ``trace_id``: a second rating for
         *     the same answer returns the existing record instead of appending a
         *     duplicate. The feedback is anchored to the unanswered queue when a queued
         *     item exists for the same ``trace_id``.
         *
         *     Malformed input (empty/oversized ``trace_id``, oversized ``reason``) is
         *     refused with 400 — the queue service's ``ValueError`` contract — rather
         *     than surfacing as an unhandled 500. A malformed request must never
         *     register as a server fault.
         */
        post: operations["feedback_v2_feedback_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v2/graph/stats": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Graph Stats
         * @description Return current knowledge graph statistics.
         */
        get: operations["graph_stats_v2_graph_stats_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v2/graph/update": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Incremental Graph Update
         * @description Incrementally add new chunks to the knowledge graph.
         *
         *     Request body: {"chunks": [{"id": "...", "text": "...", "faq_type": "..."}]}
         *
         *     Enables 10-20 year scalability: add new admission data without full rebuild.
         */
        post: operations["incremental_graph_update_v2_graph_update_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v2/query": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Unified Query
         * @description Unified RAG query: auto-routes to PaddedRAG or GraphRAG.
         *
         *     - Simple 1-hop queries → PaddedRAG (faster, ~150ms)
         *     - Multi-hop / comparative → GraphRAG (+PPR, ~400ms)
         *
         *     Routing is performed by HyDE + Step-Back classification using gemini-2.5-flash.
         */
        post: operations["unified_query_v2_query_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v2/query/stream": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Unified Query Stream
         * @description SSE true-token-stream variant of /v2/query.
         *
         *     Stage events → meta → token deltas → done (guarded). FE reconciles
         *     on done if deltas differ from guarded answer.
         */
        post: operations["unified_query_stream_v2_query_stream_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v2/recommend": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Post Recommend
         * @description Rank HUSC majors by a candidate's (score + ưu_tiên) vs điểm chuẩn.
         *
         *     Returns a deterministic, pure-arithmetic ranking — no LLM call, no
         *     network, no DB. Response time is single-digit ms.
         */
        post: operations["post_recommend_v2_recommend_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v2/whatif": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Post Whatif
         * @description Return a band-based probability estimate for one specific major.
         *
         *     Unknown major_code → graceful BLOCKED-style payload (band=unknown_major,
         *     basis/disclaimer surfaced, NEVER 500).
         */
        post: operations["post_whatif_v2_whatif_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v3/query": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * V3 Query
         * @description V3 query: Hybrid (Dense+BM25 RRF) + Booster + Reranker + Generator.
         *
         *     Flow:
         *       1. Embedding service: encode query
         *       2. HybridSearchService.retrieve(top_k=candidate_pool) → fused docs
         *          (fallback dense-only nếu hybrid_search_service is None)
         *       3. boost_with_aggregation(query, docs, max_inject=2)
         *       4. reranker.rerank(query, chunks, top_k, apply_lost_in_middle)
         *       5. llm_generator.generate_answer(query, chunks, confidence, is_enum)
         */
        post: operations["v3_query_v3_query_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/warmup": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Warmup Endpoint
         * @description Preload the embedding model to remove the cold-start penalty from
         *     the user's first real query. Returns 503 if the service is not yet
         *     initialized; otherwise 200 with the warmup latency in ms.
         */
        get: operations["warmup_endpoint_warmup_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
};
export type webhooks = Record<string, never>;
export type components = {
    schemas: {
        /**
         * AdminIdentityResponse
         * @description The caller's own identity plus the actions its role grants.
         */
        AdminIdentityResponse: {
            /** Actions */
            actions: string[];
            /** Actor Id */
            actor_id: string;
            /** Role */
            role: string;
        };
        /**
         * AuditAction
         * @description What happened. An action outside this enum is not recordable.
         * @enum {string}
         */
        AuditAction: "publish" | "rollback" | "permission_change" | "source_document_change" | "conflict_override" | "create" | "edit" | "submit" | "approve" | "reject" | "supersede" | "withdraw" | "revoke_supersession" | "claim";
        /**
         * AuditEventList
         * @description Newest-first page of the ledger.
         */
        AuditEventList: {
            /** Count */
            count: number;
            /** Events */
            events: components["schemas"]["AuditEventRecord"][];
        };
        /**
         * AuditEventRecord
         * @description One stored event plus its position in the hash chain.
         *
         *     The hashes are part of the payload on purpose: ``verify`` is only meaningful
         *     to a caller who can also see what it verified, and an auditor comparing this
         *     listing against an archived copy needs the links, not just the events.
         */
        AuditEventRecord: {
            /** Entry Hash */
            entry_hash: string;
            /** Event */
            event: {
                [key: string]: unknown;
            };
            /** Prev Hash */
            prev_hash: string;
            /** Seq */
            seq: number;
        };
        /**
         * AuthorityLevel
         * @description Source authority tier of the fact.
         *
         *     Ordered by the spec's proposed source priority ("Thứ tự ưu tiên nguồn"),
         *     highest first. That order still needs HUSC sign-off, so it lives in exactly
         *     one place (``_AUTHORITY_RANK``) instead of being re-derived per caller.
         * @enum {string}
         */
        AuthorityLevel: "husc_canonical" | "husc_official" | "hue_university" | "ministry" | "prior_year_official" | "derived_faq";
        /**
         * BackfillView
         * @description What a backfill over the record store did, with the counts measured.
         */
        BackfillView: {
            /** Already Registered */
            already_registered: string[];
            /** Citations */
            citations: number;
            /** File Absent */
            file_absent: string[];
            /** Registered */
            registered: string[];
            /** Scanned Versions */
            scanned_versions: number;
            /** Skipped Kinds */
            skipped_kinds: {
                [key: string]: number;
            };
        };
        /** BlindCitationItemView */
        BlindCitationItemView: {
            /** Kind */
            kind: string;
            /** Record Id */
            record_id: string;
            /** Record Version Key */
            record_version_key: string;
            /** Source Document Id */
            source_document_id: string;
            /** Status */
            status: string;
            /** Version */
            version: number;
        };
        /**
         * BlindCitationListView
         * @description The named state: record versions citing something unregistered.
         *
         *     ``items`` being non-empty is the expected reading today (the store's
         *     ``*.jsonl#slug`` pointers are here by design); the observable the ticket
         *     asks for is that the list exists and enumerates, not that it is clean.
         */
        BlindCitationListView: {
            /** Cited Versions */
            cited_versions: number;
            /** Counts By Document Id */
            counts_by_document_id: {
                [key: string]: number;
            };
            /** Counts By Kind */
            counts_by_kind: {
                [key: string]: number;
            };
            /** Items */
            items: components["schemas"]["BlindCitationItemView"][];
            /** Scanned Versions */
            scanned_versions: number;
            /** Statuses */
            statuses: string[];
            /** Unregistered Count */
            unregistered_count: number;
        };
        /** Body_upload_source_document_admin_source_documents_upload_post */
        Body_upload_source_document_admin_source_documents_upload_post: {
            /** File */
            file: string;
        };
        /**
         * CanonicalRecord
         * @description One versioned admissions fact: effective window + provenance + text.
         *
         *     A record cannot exist with a reversed effective window, and a ``published``
         *     record cannot exist without a source document. Both are domain invariants
         *     enforced at the boundary, not in a retrieval prompt.
         *
         *     ``source_document_id`` / ``source_url`` / ``source_page`` are nullable so a
         *     draft can be authored before its source document is uploaded; only the
         *     published state requires provenance to be present.
         */
        CanonicalRecord: {
            /** Admission Methods */
            admission_methods?: string[];
            /** Admission Round */
            admission_round?: string | null;
            /** Admission Year */
            admission_year: number;
            /** Authored By */
            authored_by?: string | null;
            authority_level: components["schemas"]["AuthorityLevel"];
            /**
             * Effective From
             * Format: date
             */
            effective_from: string;
            /** Effective To */
            effective_to?: string | null;
            /**
             * Locale
             * @default vi
             * @constant
             */
            locale: "vi";
            /** Major Codes */
            major_codes?: string[];
            /** Published At */
            published_at?: string | null;
            /** Record Id */
            record_id: string;
            record_type: components["schemas"]["RecordType"];
            /**
             * Schema Version
             * @default 1.1
             */
            schema_version: string;
            /** Source Document Id */
            source_document_id?: string | null;
            /** Source Page */
            source_page?: number | null;
            /** Source Url */
            source_url?: string | null;
            status: components["schemas"]["RecordStatus"];
            /** Supersedes */
            supersedes?: string | null;
            /** Text */
            text: string;
            /** Version */
            version: number;
        } & {
            [key: string]: unknown;
        };
        /**
         * CellProvenance
         * @description Where one cell's value came from.
         */
        CellProvenance: {
            /**
             * Chunk Id
             * @description Retrieval chunk that materialized the value, when known
             */
            chunk_id?: string | null;
            /**
             * Record Version Key
             * @description `record_id@version` of the canonical record version
             */
            record_version_key: string;
            /**
             * Source Document Id
             * @description Repo-relative document id the record was extracted from
             */
            source_document_id?: string | null;
            /**
             * Source Page
             * @description 1-based page in the source document; None when unknown
             */
            source_page?: number | null;
            /**
             * Source Url
             * @description Canonical source URL when available
             */
            source_url?: string | null;
        };
        /**
         * ChainVerificationResponse
         * @description The result of walking the chain.
         */
        ChainVerificationResponse: {
            /** Broken Index */
            broken_index?: number | null;
            /** Checked */
            checked: number;
            /** Last Hash */
            last_hash?: string | null;
            /** Ok */
            ok: boolean;
            /** Reason */
            reason?: string | null;
        };
        /**
         * ChunkPreview
         * @description Individual chunk preview
         */
        ChunkPreview: {
            /**
             * Char Count
             * @description Character count
             */
            char_count: number;
            /**
             * Chunk Id
             * @description Chunk index
             */
            chunk_id: number;
            /**
             * Sparse Terms Count
             * @description Number of BM25 sparse terms extracted
             */
            sparse_terms_count: number;
            /**
             * Text
             * @description Chunk text
             */
            text: string;
            /**
             * Token Count
             * @description Token count for this chunk
             */
            token_count: number;
        };
        /**
         * ChunkPreviewRequest
         * @description Chunk preview request model
         */
        ChunkPreviewRequest: {
            /**
             * Profile
             * @description Chunking profile to use
             * @default auto
             * @example auto
             */
            profile: string;
            /**
             * Text
             * @description Text to chunk (minimum 10 characters)
             * @example Trong công tác tuyển sinh năm 2025, Bộ GDĐT quy định tất cả thí sinh phải đăng ký xét tuyển trực tuyến.
             */
            text: string;
        };
        /**
         * ChunkPreviewResponse
         * @description Chunk preview response model
         */
        ChunkPreviewResponse: {
            /**
             * Avg Tokens Per Chunk
             * @description Average tokens per chunk
             */
            avg_tokens_per_chunk: number;
            /**
             * Chunks
             * @description List of chunk previews
             */
            chunks: components["schemas"]["ChunkPreview"][];
            /**
             * Profile Config
             * @description Profile configuration details
             */
            profile_config: {
                [key: string]: unknown;
            };
            /**
             * Profile Used
             * @description Chunking profile used
             */
            profile_used: string;
            /**
             * Text Length
             * @description Original text length in characters
             */
            text_length: number;
            /**
             * Total Chunks
             * @description Total number of chunks generated
             */
            total_chunks: number;
            /**
             * Total Tokens
             * @description Sum of tokens across all chunks
             */
            total_tokens: number;
        };
        /**
         * ClaimCitation
         * @description One high-impact claim traced to its source.
         *
         *     ``grounded`` is False when no retrieved chunk contains the claim's value;
         *     abstention policy (ticket 11) consumes that flag -- it is the only field
         *     allowed to be False without invalidating the response.
         */
        ClaimCitation: {
            /**
             * Chunk Id
             * @description Chunk that grounds the claim
             */
            chunk_id?: string | null;
            /**
             * Claim
             * @description Exact claim text as it appears in the answer
             */
            claim: string;
            /**
             * Claim Type
             * @description High-impact category of the claim
             * @enum {string}
             */
            claim_type: "cutoff_score" | "timeline" | "quota" | "tuition" | "other";
            /**
             * Grounded
             * @description False means the claim value was not found in any retrieved chunk
             * @default false
             */
            grounded: boolean;
            /**
             * Note
             * @description Provenance caveat, e.g. the rowspan_unresolved reason
             */
            note?: string | null;
            /**
             * Record Id
             * @description Canonical record the chunk materialized from
             */
            record_id?: string | null;
            /**
             * Record Version Key
             * @description Exact record version (release-scoped identity)
             */
            record_version_key?: string | null;
            /**
             * Row Label
             * @description Identity label of the table row the claim sits in (resolved across split pieces)
             */
            row_label?: string | null;
            /**
             * Section
             * @description Section/breadcrumb label inside the source document, when present
             */
            section?: string | null;
            /**
             * Source Document Id
             * @description Source document the record was extracted from
             */
            source_document_id?: string | null;
            /**
             * Source Page
             * @description Page in the source document; None when unknown -- never fabricated
             */
            source_page?: number | null;
            /**
             * Source Url
             * @description Canonical source URL when available
             */
            source_url?: string | null;
            /**
             * Value
             * @description Normalized value token (digits, decimalized)
             */
            value: string;
        };
        /**
         * ClaimRequest
         * @description Body for POST /admin/records/{record_id}/claim.
         */
        ClaimRequest: {
            /** Expected Version */
            expected_version?: number | null;
            /** Reason */
            reason?: string | null;
        };
        /**
         * CompareRequest
         * @description A multi-major same-year comparison question.
         */
        CompareRequest: {
            /** Effective Year */
            effective_year: number;
            /**
             * Intent
             * @default so_sanh
             */
            intent: string;
            /** Query */
            query: string;
            /**
             * Record Ids
             * @description Candidate major record_ids to compare
             */
            record_ids?: string[];
            /**
             * Route
             * @default graph_rag
             */
            route: string;
        };
        /**
         * ComparisonCell
         * @description One major x metric cell of the comparison table.
         */
        ComparisonCell: {
            /**
             * Major
             * @description Major label this cell belongs to
             */
            major: string;
            /**
             * Metric
             * @description Metric this cell reports
             */
            metric: string;
            /**
             * Note
             * @description Observed caveat, e.g. 'số liệu ước đạo'
             */
            note?: string | null;
            /** @description Provenance of the value; None when missing */
            source?: components["schemas"]["CellProvenance"] | null;
            /**
             * State
             * @description `present` or `missing`
             */
            state: string;
            /**
             * Value
             * @description Rendered value; None when missing
             */
            value?: string | null;
        };
        /**
         * ComparisonColumn
         * @description One metric column header of the table.
         */
        ComparisonColumn: {
            /**
             * Display
             * @description Human-readable column header
             */
            display: string;
            /**
             * Metric
             * @description Field label, e.g. 'điểm chuẩn' or 'học phí'
             */
            metric: string;
        };
        /**
         * ComparisonResponse
         * @description Top-level comparison answer returned by /v2/compare.
         *
         *     Exactly one primary ``table`` (the same-year group); majors that could not
         *     be grouped (different effective year) are listed in ``refused`` so the FE
         *     can surface them instead of silently dropping or guessing. The
         *     ``route``/``intent`` echo the router's decision so the client can confirm
         *     the question was served by the multihop path (rule 4).
         */
        ComparisonResponse: {
            /**
             * Effective Year
             * @description Year all table rows share
             */
            effective_year: number;
            /**
             * Intent
             * @description Router intent label
             */
            intent: string;
            /**
             * Query
             * @description The comparison question answered
             */
            query: string;
            /** Refused */
            refused?: components["schemas"]["RefusedMajor"][];
            /**
             * Route
             * @description Pipeline route — must be 'graph_rag' for a comparison
             */
            route: string;
            table: components["schemas"]["ComparisonTable"];
        };
        /**
         * ComparisonRow
         * @description One major's row of cells.
         */
        ComparisonRow: {
            /** Cells */
            cells?: components["schemas"]["ComparisonCell"][];
            /**
             * Effective Year
             * @description Admission year this row's facts carry
             */
            effective_year: number;
            /**
             * Major
             * @description Major label, matches its cells
             */
            major: string;
            /**
             * Record Version Key
             * @description Latest record version backing this major
             */
            record_version_key: string;
        };
        /**
         * ComparisonTable
         * @description The comparison answer: one table per effective year.
         *
         *     Rule 1 lives in :meth:`assert_year_uniformity`: every row MUST share the
         *     table's ``effective_year``, otherwise the table is invalid. The service
         *     splits differing-year majors into ``refused`` instead of producing such a
         *     table, so a constructed table is always year-uniform — but the validator
         *     keeps that invariant true even for hand-built inputs (the contract is the
         *     source of truth, not the service).
         */
        ComparisonTable: {
            /** Columns */
            columns: components["schemas"]["ComparisonColumn"][];
            /**
             * Effective Year
             * @description Admission year all rows share
             */
            effective_year: number;
            /**
             * Query
             * @description The comparison question this table answers
             */
            query: string;
            /**
             * Refused
             * @description Majors excluded because their year did not match
             */
            refused?: components["schemas"]["RefusedMajor"][];
            /** Rows */
            rows?: components["schemas"]["ComparisonRow"][];
        };
        /**
         * ConflictItemView
         * @description One queued conflict, as an officer reads it before deciding.
         */
        ConflictItemView: {
            /** Admission Year */
            admission_year: number;
            /** Conflict Id */
            conflict_id: string;
            /** Field */
            field: string;
            /** Ma Nganh */
            ma_nganh: string;
            /** Origin */
            origin: string;
            /** Record Type */
            record_type: string;
            /** Resolution */
            resolution?: {
                [key: string]: unknown;
            } | null;
            /** Statements */
            statements?: {
                [key: string]: unknown;
            };
            /** Status */
            status: string;
        };
        /**
         * ConflictQueueResponse
         * @description The queue plus the accounting the ticket's criteria measure.
         *
         *     ``seed_reported_items`` counts items whose origin is the seed report, so
         *     "all 74 entered the queue" is checkable without parsing the item list;
         *     ``deduped`` / ``dedup_reason`` state, per criterion 9, what the detector
         *     collapsed and why instead of hiding it.
         */
        ConflictQueueResponse: {
            /** Dedup Reason */
            dedup_reason?: string | null;
            /** Deduped */
            deduped: number;
            /** Items */
            items: components["schemas"]["ConflictItemView"][];
            /** Open */
            open: number;
            /** Resolved */
            resolved: number;
            /** Seed Reported Items */
            seed_reported_items: number;
            /** Total */
            total: number;
        };
        /**
         * CreateRecordRequest
         * @description Body for POST /admin/records — a full canonical record, version 1.
         *
         *     No ``actor_id``: the actor is whoever the credential resolves to, so a
         *     caller cannot write a version under someone else's name.
         */
        CreateRecordRequest: {
            record: components["schemas"]["CanonicalRecord"];
        };
        /**
         * EditRecordRequest
         * @description Body for POST /admin/records/{record_id}/edits.
         *
         *     Only the listed fields are editable: identity (``record_id``), the
         *     ``version`` counter and lifecycle ``status`` are owned by the store and the
         *     state machine, never by a caller. ``expected_version`` is the optimistic
         *     concurrency token — the version the caller believes is current.
         *
         *     ``record_type`` is absent on purpose: the record's kind is part of what the
         *     fact *is*, and the id embeds it (``CHI_TIEU:nganh:...``), so re-typing a
         *     fact is a new record rather than an edit of this one.
         *
         *     ``values`` is the record's structured payload (the canonical schema is
         *     additive, ``extra="allow"``). It **replaces** the payload wholesale, like
         *     every other field here; a caller that edits one structured field merges the
         *     rest itself. The admin UI does exactly that, so the extractor's other keys
         *     survive a hand edit.
         */
        EditRecordRequest: {
            /** Admission Methods */
            admission_methods?: string[] | null;
            /** Admission Round */
            admission_round?: string | null;
            /** Admission Year */
            admission_year?: number | null;
            authority_level?: components["schemas"]["AuthorityLevel"] | null;
            /** Effective From */
            effective_from?: string | null;
            /** Effective To */
            effective_to?: string | null;
            /** Expected Version */
            expected_version: number;
            /** Major Codes */
            major_codes?: string[] | null;
            /** Reason */
            reason?: string | null;
            /** Source Document Id */
            source_document_id?: string | null;
            /** Source Page */
            source_page?: number | null;
            /** Source Url */
            source_url?: string | null;
            /** Supersedes */
            supersedes?: string | null;
            /** Text */
            text?: string | null;
            /** Values */
            values?: {
                [key: string]: unknown;
            } | null;
        };
        /**
         * FeedbackRequest
         * @description Feedback on a served answer (public endpoint for the student).
         */
        FeedbackRequest: {
            /**
             * Helpful
             * @description True for 👍, False for 👎
             */
            helpful: boolean;
            /**
             * Question
             * @description The question that produced this answer, when the client has it
             */
            question?: string | null;
            /**
             * Reason
             * @description Optional free-text reason
             */
            reason?: string | null;
            /**
             * Trace Id
             * @description trace_id of the answer being rated
             */
            trace_id: string;
        };
        /**
         * FeedbackResponse
         * @description What the feedback call recorded.
         */
        FeedbackResponse: {
            /** Helpful */
            helpful: boolean;
            /** Reason */
            reason?: string | null;
            /** Recorded */
            recorded: boolean;
            /** Trace Id */
            trace_id: string;
        };
        /** HTTPValidationError */
        HTTPValidationError: {
            /** Detail */
            detail?: components["schemas"]["ValidationError"][];
        };
        /**
         * HealthResponse
         * @description Health check response.
         *
         *     Backward-compatible: legacy fields remain so older clients (FE
         *     pre-Phase-1) keep working. New fields drive the `ready`/`stage`
         *     contract that the FE polls to gate Send-button activation.
         */
        HealthResponse: {
            /**
             * Collection
             * @default
             */
            collection: string;
            /**
             * Embedding Dim
             * @default -1
             */
            embedding_dim: number;
            /**
             * Embedding Model
             * @default
             */
            embedding_model: string;
            /**
             * Lancedb Connected
             * @default false
             */
            lancedb_connected: boolean;
            /** Metrics */
            metrics?: {
                [key: string]: unknown;
            };
            /**
             * Ready
             * @default false
             */
            ready: boolean;
            /**
             * Reranker Load Ok
             * @default false
             */
            reranker_load_ok: boolean;
            /**
             * Reranker Model
             * @default
             */
            reranker_model: string;
            /** Services */
            services?: {
                [key: string]: boolean;
            };
            /**
             * Stage
             * @default warming
             * @enum {string}
             */
            stage: "warming" | "ready" | "failed";
            /** Status */
            status: string;
            /**
             * Vector Dim Match
             * @default false
             */
            vector_dim_match: boolean;
            /**
             * Vectors Count
             * @default 0
             */
            vectors_count: number;
        };
        /**
         * IdentityCreateRequest
         * @description Body for ``POST /admin/identities``.
         *
         *     The credential is supplied by the caller rather than generated here: this
         *     surface hands an existing secret to a named person, and inventing one in a
         *     response body would put a live credential in a place logs read.
         */
        IdentityCreateRequest: {
            /** Actor Id */
            actor_id: string;
            /** Credential */
            credential: string;
            /** Password */
            password?: string | null;
            /** Reason */
            reason: string;
            role: components["schemas"]["Role"];
            /** Subject Id */
            subject_id?: string | null;
        };
        /**
         * IdentityPatchRequest
         * @description Body for ``PATCH /admin/identities/{subject_id}``.
         *
         *     ``role`` and ``disabled`` are both optional but at least one is required -
         *     a mutation that changes nothing would still append an event, and an audit
         *     trail with no-ops in it stops being evidence.
         */
        IdentityPatchRequest: {
            /** Disabled */
            disabled?: boolean | null;
            /** Password */
            password?: string | null;
            /** Reason */
            reason: string;
            role?: components["schemas"]["Role"] | null;
        };
        /**
         * IdentityView
         * @description One stored identity, with its credential masked.
         *
         *     ``subject_id`` is exposed so the admin surface has a stable key to PATCH;
         *     ``actor_id`` can be renamed by an office and is not a URL segment.
         */
        IdentityView: {
            /** Actor Id */
            actor_id: string;
            /** Credential */
            credential: string;
            /** Disabled */
            disabled: boolean;
            role: components["schemas"]["Role"];
            /** Subject Id */
            subject_id: string;
        };
        /**
         * IndexJobResponse
         * @description One index-job row.
         *
         *     ``state`` is a JobState value (pending/running/failed/active) or null when
         *     the row is a build on disk rather than a run. ``saved`` and ``serving`` are
         *     separate on purpose: ``saved`` means a build is persisted, ``serving`` means
         *     it is the corpus the index answers from, and only ``active`` sets ``serving``.
         */
        IndexJobResponse: {
            /** Job Id */
            job_id: string;
            /** Label */
            label: string;
            /** Reason */
            reason: string | null;
            /** Release Id */
            release_id: string | null;
            /** Retry */
            retry: components["schemas"]["RetryStepResponse"][];
            /** Saved */
            saved: boolean;
            /** Serving */
            serving: boolean;
            /** Staged Dir */
            staged_dir: string | null;
            /** State */
            state: string | null;
        };
        /**
         * IndexJobsResponse
         * @description GET /admin/index-jobs — job state, staged builds, and what is owed.
         */
        IndexJobsResponse: {
            /** Jobs */
            jobs: components["schemas"]["IndexJobResponse"][];
            /** Pending Affected Record Ids */
            pending_affected_record_ids: string[];
            /** Pending Outbox Entries */
            pending_outbox_entries: number;
            /** Problems */
            problems: string[];
            /** Reindex Needed */
            reindex_needed: boolean;
            /** Reindex Reason */
            reindex_reason: string;
        };
        /**
         * LifecycleAction
         * @description A requested move along the lifecycle. Not every action is legal in
         *     every state — see ``_TRANSITIONS``.
         * @enum {string}
         */
        LifecycleAction: "submit" | "approve" | "reject" | "publish" | "supersede" | "withdraw" | "revoke_supersession";
        /**
         * LifecycleRecordResponse
         * @description The stored record version plus the audit event the write emitted.
         */
        LifecycleRecordResponse: {
            /** Event */
            event: {
                [key: string]: unknown;
            };
            record: components["schemas"]["CanonicalRecord"];
        };
        /** LoginRequest */
        LoginRequest: {
            /** Actor Id */
            actor_id: string;
            /** Password */
            password: string;
        };
        /** LoginResponse */
        LoginResponse: {
            /** Actor Id */
            actor_id: string;
            /** Role */
            role: string;
            /**
             * Session Seconds
             * @default 28800
             */
            session_seconds: number;
        };
        /**
         * MetaResponse
         * @description Public payload for /api/meta.
         */
        MetaResponse: {
            /** Current Admission Year */
            current_admission_year: number;
            /** Freshness Alert */
            freshness_alert: boolean;
            /** Freshness Lag Days */
            freshness_lag_days: number | null;
            /** Latest Crawl Date */
            latest_crawl_date: string | null;
            /** Total Notifications */
            total_notifications: number | null;
        };
        /** PreviewRequest */
        PreviewRequest: {
            /** Query */
            query: string;
            /**
             * Top K
             * @default 5
             */
            top_k: number;
            /** Version */
            version?: number | null;
        };
        /**
         * QueryResponse
         * @description Enhanced query response with all fields
         */
        QueryResponse: {
            /** Answer */
            answer: string;
            /** Chunks */
            chunks?: {
                [key: string]: unknown;
            }[] | null;
            /** Chunks Used */
            chunks_used: number;
            /** Confidence */
            confidence: number;
            /** Data Gap Hints */
            data_gap_hints?: string[];
            /** Enhanced Query */
            enhanced_query: string;
            /**
             * Groundedness Score
             * @default 0
             */
            groundedness_score: number;
            /** Internal Status Code */
            internal_status_code?: string | null;
            /** Original Query */
            original_query: string;
            /**
             * Pii Detected
             * @default false
             */
            pii_detected: boolean;
            /** Provider */
            provider: string;
            /** Query Type */
            query_type: string;
            /** Sources */
            sources: components["schemas"]["SourceChip"][];
            /**
             * Status Code
             * @default SUCCESS
             */
            status_code: string;
            /**
             * Status Reason
             * @default
             */
            status_reason: string;
            /** Top K Used */
            top_k_used: number;
            /** Trace Id */
            trace_id: string;
        };
        /** RecommendRequest */
        RecommendRequest: {
            /**
             * Score
             * @description Tổng điểm thang 30
             */
            score: number;
            /**
             * To Hop
             * @description Tổ hợp môn xét tuyển, vd 'A00'. Optional.
             */
            to_hop?: string | null;
            /**
             * Uu Tien
             * @description Điểm ưu tiên (khu vực / đối tượng / chứng chỉ). Mặc định 0.
             * @default 0
             */
            uu_tien: number;
        };
        /** RecommendResponse */
        RecommendResponse: {
            /** Basis */
            basis: string;
            /** Count */
            count: number;
            /** Disclaimer */
            disclaimer: string;
            /** Recommendations */
            recommendations: components["schemas"]["RecommendationOut"][];
            /** Score */
            score: number;
            /** To Hop */
            to_hop?: string | null;
            /** Uu Tien */
            uu_tien: number;
        };
        /** RecommendationOut */
        RecommendationOut: {
            /** Delta */
            delta: number;
            /** Explanation */
            explanation: string;
            /** Label */
            label: string;
            /** Latest Diem Chuan */
            latest_diem_chuan: number;
            /** Latest Year */
            latest_year: number;
            /** Major Code */
            major_code: string;
            /** Major Name */
            major_name: string;
            /** To Hop */
            to_hop?: string[] | null;
        };
        /**
         * RecordHistoryResponse
         * @description Every stored version of one record, newest first.
         */
        RecordHistoryResponse: {
            /** Record Id */
            record_id: string;
            /** Versions */
            versions: components["schemas"]["CanonicalRecord"][];
        };
        /**
         * RecordListResponse
         * @description Latest version of every stored record that matched the filter.
         */
        RecordListResponse: {
            /** Records */
            records: components["schemas"]["CanonicalRecord"][];
            /** Total */
            total: number;
        };
        /**
         * RecordStatus
         * @description Lifecycle state of a record version (spec state machine, verbatim).
         *
         *     ``draft --submit--> review --approve--> approved --publish--> published
         *     --supersede--> archived``.
         * @enum {string}
         */
        RecordStatus: "draft" | "review" | "approved" | "published" | "archived";
        /**
         * RecordType
         * @description Kind of admissions fact a canonical record states.
         *
         *     First group: mirrors ``EntityType`` values verbatim (shared vocabulary).
         *     Second group: canonical-only fact kinds (no graph entity counterpart).
         *
         *     **Rollback constraint (1.1, ticket 03):** ``THONG_BAO`` and ``HOI_DAP``
         *     are 1.1 additions. Once a store contains either value, reverting the
         *     reader to 1.0 code will raise ``ValidationError`` on load. A code
         *     rollback therefore requires restoring a store snapshot taken *before*
         *     1.1 was activated (spec 02 §2); a release-only rollback (keep code,
         *     point back at an older release) is always safe.
         * @enum {string}
         */
        RecordType: "NGANH" | "TO_HOP" | "DIEM_CHUAN" | "HOC_PHI" | "THOI_GIAN" | "TO_CHUC" | "CHINH_SACH" | "THONG_BAO" | "HOI_DAP" | "CHI_TIEU" | "DIEU_KIEN";
        /**
         * RefusedMajor
         * @description A major split out of the table because its year did not match.
         */
        RefusedMajor: {
            /**
             * Effective Year
             * @description The year the major actually carries
             */
            effective_year: number;
            /**
             * Major
             * @description Major label being refused
             */
            major: string;
            /**
             * Reason
             * @description Why the major was excluded from the table
             */
            reason: string;
        };
        /**
         * RegisteredCheckView
         * @description The answer to "is this document id in the registry?" plus why.
         *
         *     ``kind`` is returned even when unregistered, so a caller sees whether the
         *     id was a corpus pointer (never registerable here) or a document nobody
         *     registered yet — the two read very differently to an admissions officer.
         */
        RegisteredCheckView: {
            /** Document Id */
            document_id: string;
            entry?: components["schemas"]["SourceDocumentView"] | null;
            /** Kind */
            kind: string;
            /** Registered */
            registered: boolean;
        };
        /**
         * RegistrationRequest
         * @description The body of a register/update call: the id and an optional note.
         */
        RegistrationRequest: {
            /** Document Id */
            document_id: string;
            /** Notes */
            notes?: string | null;
        };
        /**
         * ReleaseStatusResponse
         * @description GET /admin/releases — the active release and the one behind it.
         *
         *     ``active_release_id`` is what the serving path reads (the pointer) and is
         *     present even when the release it names cannot be read; ``active``/``previous``
         *     are the manifest-backed views. ``problems`` carries anything that could not
         *     be read, so an unreadable pointer is never rendered as "nothing published".
         */
        ReleaseStatusResponse: {
            active: components["schemas"]["ReleaseViewResponse"] | null;
            /** Active Release Id */
            active_release_id: string | null;
            previous: components["schemas"]["ReleaseViewResponse"] | null;
            /** Previous Release Id */
            previous_release_id: string | null;
            /** Problems */
            problems: string[];
        };
        /**
         * ReleaseViewResponse
         * @description One release, as its own manifest states it.
         */
        ReleaseViewResponse: {
            /** Chunk Count */
            chunk_count: number;
            /** Created At */
            created_at: string | null;
            /** Document Count */
            document_count: number;
            /** Lancedb Tag */
            lancedb_tag: string | null;
            /** Lancedb Version */
            lancedb_version: number | null;
            /** Parent Release Id */
            parent_release_id: string | null;
            /** Release Id */
            release_id: string;
        };
        /**
         * ReloadTableRequest
         * @description Body for POST /admin/reload-table.
         *
         *     ``force_while_in_flight`` is the emergency override for the drain gate, and
         *     is named for exactly what it does: swap while requests are still in flight,
         *     accepting that a request already reading the old retriever fails. It is a
         *     second, explicit decision by the caller — never a meaning read into the
         *     drain timeout. A plain reload whose drain times out is refused with 409.
         */
        ReloadTableRequest: {
            /**
             * Drain Timeout S
             * @default 25
             */
            drain_timeout_s: number;
            /**
             * Force While In Flight
             * @description Emergency override: swap even though the drain timed out. A request already reading the old retriever may then fail; the response reports ``forced_while_in_flight``.
             * @default false
             */
            force_while_in_flight: boolean;
            /** Table Name */
            table_name: string;
        };
        /**
         * ReloadTableResponse
         * @description Response for POST /admin/reload-table.
         *
         *     ``drained`` is always True on a 200: a drain that times out is a 409 unless
         *     the caller forced it, in which case ``forced_while_in_flight`` is True.
         */
        ReloadTableResponse: {
            /** Active Table */
            active_table: string;
            /** Drain Ms */
            drain_ms: number;
            /** Drained */
            drained: boolean;
            /** Forced While In Flight */
            forced_while_in_flight: boolean;
            /** In Flight At Swap */
            in_flight_at_swap: number;
            /** Total Ms */
            total_ms: number;
        };
        /**
         * ResolveConflictRequest
         * @description The decision body: the winning source and the reason.
         *
         *     ``actor`` is deliberately absent — the event's actor comes from the
         *     credential, never from the body.
         */
        ResolveConflictRequest: {
            /** Reason */
            reason?: string | null;
            /** Winner Source Document Id */
            winner_source_document_id: string;
        };
        /**
         * RetireRequest
         * @description Retirement names a reason: it un-vouches for live citations.
         */
        RetireRequest: {
            /** Document Id */
            document_id: string;
            /** Reason */
            reason: string;
        };
        /**
         * RetryStepResponse
         * @description One step of retrying a failed job, and whether this process can run it.
         */
        RetryStepResponse: {
            /** Command */
            command: string | null;
            /** Reason */
            reason: string | null;
            /** Runnable */
            runnable: boolean;
            /** Step */
            step: string;
        };
        /**
         * Role
         * @description Who a caller is, as decided by the server from its credential.
         * @enum {string}
         */
        Role: "viewer" | "editor" | "reviewer" | "publisher" | "reviewer_publisher" | "admin" | "ops";
        /**
         * RollbackRequest
         * @description Body for POST /admin/rollback.
         *
         *     ``reason`` is mandatory and non-blank. The event the rollback emits refuses
         *     to exist without one (``AuditAction.ROLLBACK`` is in ``REASON_REQUIRED``),
         *     so a blank reason is refused here — 422, before any store, pointer or table
         *     is touched.
         *
         *     ``target_release_id`` is optional. Without it the target is the previous
         *     release in the lineage (``parent_release_id``, walked back from the active
         *     one); with it, a rollback can go further back than one release.
         *
         *     ``force_while_in_flight`` is the emergency override for the drain gate. A
         *     rollback whose drain times out is refused with 409 like a reload is; this
         *     flag is how an operator says "roll back anyway, knowing a request still
         *     reading the release I am rolling away from may fail". It is a separate,
         *     explicitly-named decision — never a meaning read into the timeout.
         */
        RollbackRequest: {
            /**
             * Force While In Flight
             * @description Emergency override: roll back even though the drain timed out. A request already reading the old release may then fail; the response reports ``forced_while_in_flight``.
             * @default false
             */
            force_while_in_flight: boolean;
            /** Reason */
            reason: string;
            /** Target Release Id */
            target_release_id?: string | null;
        };
        /**
         * RollbackResponse
         * @description What the rollback did, per level, plus the audit event it emitted.
         */
        RollbackResponse: {
            /** Event Id */
            event_id: string | null;
            /** Forced While In Flight */
            forced_while_in_flight: boolean;
            /** Lancedb Tag */
            lancedb_tag: string;
            /** Lancedb Tag Version */
            lancedb_tag_version: number | null;
            /** Lancedb Version */
            lancedb_version: number | null;
            /** Previous Release Id */
            previous_release_id: string;
            /** Release Id */
            release_id: string;
            /** Steps */
            steps: {
                [key: string]: string;
            };
        };
        /**
         * SimpleQueryRequest
         * @description Simple query request - user only needs to send a string
         */
        SimpleQueryRequest: {
            /**
             * Force Rag Only
             * @description Force RAG only mode
             * @default false
             */
            force_rag_only: boolean | null;
            /**
             * Query
             * @description User's question
             */
            query: string;
        };
        /**
         * SourceChip
         * @description Structured source citation for FE chip rendering (rich-markdown T4).
         */
        SourceChip: {
            /**
             * Data Year
             * @description Effective year of the data (e.g. 2025, 2026)
             * @default N/A
             */
            data_year: string;
            /**
             * Id
             * @description Unique chunk or source identifier
             */
            id: string;
            /**
             * Snippet
             * @description Short excerpt from the chunk (first 120 chars)
             * @default
             */
            snippet: string;
            /**
             * Title
             * @description Human-readable source title (first 80 chars)
             */
            title: string;
            /**
             * Url
             * @description Canonical source URL when available
             */
            url?: string | null;
        };
        /**
         * SourceDocumentListView
         * @description The registry plus the counts that make it checkable at a glance.
         *
         *     ``total`` and ``registered`` differ only when something was retired;
         *     ``counts_by_kind`` is ``{"pdf_document": n}`` today, because the registry
         *     holds documents only (Ghi chú 4) — the field exists so a reader confirms
         *     that rather than assuming it.
         */
        SourceDocumentListView: {
            /** Counts By Kind */
            counts_by_kind: {
                [key: string]: number;
            };
            /** Entries */
            entries: components["schemas"]["SourceDocumentView"][];
            /** Registered */
            registered: number;
            /** Total */
            total: number;
        };
        /**
         * SourceDocumentView
         * @description One registry entry, as an operator reads it.
         */
        SourceDocumentView: {
            /** Document Id */
            document_id: string;
            /** File Present */
            file_present: boolean;
            /** Kind */
            kind: string;
            /** Notes */
            notes: string | null;
            /** Path */
            path: string;
            /** Registered At */
            registered_at: string;
            /** Registered By */
            registered_by: string;
            /** Revision */
            revision: number;
            /** Sha256 */
            sha256: string | null;
            /** State */
            state: string;
            /** Updated At */
            updated_at: string;
            /** Updated By */
            updated_by: string;
        };
        /**
         * TransitionRequest
         * @description Body for POST /admin/records/{record_id}/transitions.
         */
        TransitionRequest: {
            action: components["schemas"]["LifecycleAction"];
            /** Expected Version */
            expected_version?: number | null;
            /** Reason */
            reason?: string | null;
        };
        /**
         * UnansweredItem
         * @description One unanswered question, already redacted, with its resolution state.
         */
        UnansweredItem: {
            /** Asked At */
            asked_at: string;
            /** Question */
            question: string;
            /** Resolved */
            resolved: boolean;
            /** Resolved At */
            resolved_at?: string | null;
            /** Resolved By Record Id */
            resolved_by_record_id?: string | null;
            /** Route */
            route: string;
            /** Status Code */
            status_code: string;
            /** Trace Id */
            trace_id: string;
        };
        /**
         * UnansweredList
         * @description A page of unanswered items and the total count.
         */
        UnansweredList: {
            /** Count */
            count: number;
            /** Items */
            items: components["schemas"]["UnansweredItem"][];
        };
        /**
         * UnifiedQueryRequest
         * @description Unified query request for PaddedRAG + GraphRAG pipeline.
         */
        UnifiedQueryRequest: {
            /**
             * Force Route
             * @description Force routing: 'padded_rag' or 'graph_rag' (overrides auto-router)
             */
            force_route?: ("padded_rag" | "graph_rag") | null;
            /**
             * History
             * @description Optional bounded chat history, e.g. [{'role':'user','content':'...'}, ...] (max ~4 msgs)
             */
            history?: {
                [key: string]: string;
            }[] | null;
            /**
             * Query
             * @description User question (Vietnamese)
             */
            query: string;
            /**
             * Top K
             * @description Number of results
             * @default 5
             */
            top_k: number;
        };
        /**
         * UnifiedQueryResponse
         * @description Response from unified pipeline with routing metadata.
         */
        UnifiedQueryResponse: {
            /** Answer */
            answer: string;
            /** Answer Confidence */
            answer_confidence?: number | null;
            /** Citations */
            citations?: components["schemas"]["ClaimCitation"][];
            /** Confidence */
            confidence: number;
            /** Data Gap Hints */
            data_gap_hints?: string[];
            /** Graph Stats */
            graph_stats?: {
                [key: string]: unknown;
            } | null;
            /**
             * Groundedness Score
             * @default 0
             */
            groundedness_score: number;
            /** Internal Status Code */
            internal_status_code?: string | null;
            /** Latency Ms */
            latency_ms: number;
            /**
             * Pii Detected
             * @default false
             */
            pii_detected: boolean;
            /** Provider */
            provider?: string | null;
            /** Query */
            query: string;
            /** Rerank Skipped Reason */
            rerank_skipped_reason?: string | null;
            /** Route */
            route: string;
            /** Router Info */
            router_info: {
                [key: string]: unknown;
            };
            /** Sources */
            sources: components["schemas"]["SourceChip"][];
            /**
             * Status Code
             * @default SUCCESS
             */
            status_code: string;
            /** Status Reason */
            status_reason?: string | null;
            /** Trace Id */
            trace_id: string;
        };
        /**
         * UploadParseReport
         * @description The read report itself: what the parser saw, page by page.
         */
        UploadParseReport: {
            /** Chars Per Page */
            chars_per_page: number;
            /** Page Count */
            page_count: number;
            /** Pii Detected */
            pii_detected: boolean;
            /** Scan Pages */
            scan_pages: number[];
            /** Table Count */
            table_count: number;
        };
        /**
         * UploadReport
         * @description What the upload read, and what it produced.
         *
         *     ``report`` is the read report the ticket asks for (page count,
         *     characters per page, scan pages, PII flag, table count); ``warnings``
         *     is the Vietnamese operator-facing explanation of anything odd — a
         *     scan-only document, a PII flag, a read failure. ``draft_record_ids``
         *     is empty exactly when no draft was created: an image-only or
         *     unreadable document reports rather than drafts an empty record. A
         *     PDF/DOCX drafts at most one record; an XLSX drafts one per data row
         *     (ticket T17).
         */
        UploadReport: {
            /** Document Id */
            document_id: string;
            /**
             * Draft Record Id
             * @description The primary draft id — the first row's for a spreadsheet, else None.
             */
            readonly draft_record_id: string | null;
            /** Draft Record Ids */
            draft_record_ids?: string[];
            /** Kind */
            kind: string;
            report: components["schemas"]["UploadParseReport"];
            /** Sha256 */
            sha256: string;
            /** Warnings */
            warnings?: string[];
        };
        /**
         * UrlIngestReport
         * @description What the ingest did, and what it produced.
         *
         *     ``status`` is the idempotency answer: ``unchanged`` means the page's bytes
         *     matched the stored digest and *no record was written* — ``version`` is then
         *     the version already on disk.
         */
        UrlIngestReport: {
            /** Chars */
            chars: number;
            /** Record Id */
            record_id: string;
            /** Sha256 */
            sha256: string;
            /** Source Url */
            source_url: string;
            /**
             * Status
             * @enum {string}
             */
            status: "created" | "updated" | "unchanged";
            /** Version */
            version: number;
        };
        /**
         * UrlIngestRequest
         * @description Body for POST /source-documents/from-url.
         */
        UrlIngestRequest: {
            /**
             * Url
             * @description the notification page to fetch; http(s) and publicly routable only
             */
            url: string;
        };
        /** V3QueryRequest */
        V3QueryRequest: {
            /**
             * Apply Lost In Middle
             * @default true
             */
            apply_lost_in_middle: boolean;
            /**
             * Candidate Pool
             * @description Số doc lấy từ Hybrid trước khi rerank
             * @default 20
             */
            candidate_pool: number;
            /** Query */
            query: string;
            /**
             * Top K
             * @default 5
             */
            top_k: number;
        };
        /** V3QueryResponse */
        V3QueryResponse: {
            /** Answer */
            answer: string;
            /** Chunks Used */
            chunks_used: number;
            /** Confidence */
            confidence: number;
            /** Latency Ms */
            latency_ms: number;
            /** Pipeline Stages */
            pipeline_stages: {
                [key: string]: unknown;
            };
            /** Provider */
            provider: string;
            /** Query */
            query: string;
            /** Sources */
            sources: string[];
            /** Trace Id */
            trace_id: string;
        };
        /** ValidationError */
        ValidationError: {
            /** Context */
            ctx?: Record<string, never>;
            /** Input */
            input?: unknown;
            /** Location */
            loc: (string | number)[];
            /** Message */
            msg: string;
            /** Error Type */
            type: string;
        };
        /** WhatIfRequest */
        WhatIfRequest: {
            /** Major Code */
            major_code: string;
            /**
             * Score
             * @description Tổng điểm thang 30
             */
            score: number;
            /** To Hop */
            to_hop?: string | null;
            /**
             * Uu Tien
             * @default 0
             */
            uu_tien: number;
        };
        /** WhatIfResponse */
        WhatIfResponse: {
            /** Band */
            band: string;
            /** Basis */
            basis: string;
            /** Delta */
            delta?: number | null;
            /** Disclaimer */
            disclaimer: string;
            /** Latest Diem Chuan */
            latest_diem_chuan?: number | null;
            /** Latest Year */
            latest_year?: number | null;
            /** Major Code */
            major_code: string;
            /**
             * N Years
             * @default 0
             */
            n_years: number;
            /** P Pass */
            p_pass: string;
            /** Score */
            score: number;
        };
    };
    responses: never;
    parameters: never;
    requestBodies: never;
    headers: never;
    pathItems: never;
};
export type $defs = Record<string, never>;
export interface operations {
    root__get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": {
                        [key: string]: unknown;
                    };
                };
            };
        };
    };
    list_events_admin_audit_events_get: {
        parameters: {
            query?: {
                /** @description record id / release id / subject id */
                target?: string | null;
                /** @description who performed the action */
                actor_id?: string | null;
                /** @description event kind */
                action?: components["schemas"]["AuditAction"] | null;
                limit?: number;
            };
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AuditEventList"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_event_admin_audit_events__event_id__get: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path: {
                event_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AuditEventRecord"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    verify_chain_admin_audit_verify_get: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ChainVerificationResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_conflicts_admin_conflicts_get: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ConflictQueueResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    resolve_conflict_admin_conflicts__conflict_id__resolve_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path: {
                conflict_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["ResolveConflictRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ConflictItemView"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_identities_admin_identities_get: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["IdentityView"][];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    create_identity_admin_identities_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["IdentityCreateRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["IdentityView"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    patch_identity_admin_identities__subject_id__patch: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path: {
                subject_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["IdentityPatchRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["IdentityView"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_identity_admin_identity_get: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AdminIdentityResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_index_jobs_admin_index_jobs_get: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["IndexJobsResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    login_admin_login_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["LoginRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["LoginResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    logout_admin_logout_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": {
                        [key: string]: string;
                    };
                };
            };
        };
    };
    list_records_admin_records_get: {
        parameters: {
            query?: {
                /** @description Ngành: mã ngành, tên ngành, hoặc phần record_id nêu nó. */
                nganh?: string | null;
                /** @description Năm tuyển sinh. */
                year?: number | null;
                /** @description Trạng thái lifecycle của version mới nhất. */
                status?: components["schemas"]["RecordStatus"] | null;
            };
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["RecordListResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    create_record_admin_records_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["CreateRecordRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["LifecycleRecordResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_record_admin_records__record_id__get: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path: {
                record_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["CanonicalRecord"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    claim_record_admin_records__record_id__claim_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path: {
                record_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["ClaimRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["LifecycleRecordResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    edit_record_admin_records__record_id__edits_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path: {
                record_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["EditRecordRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["LifecycleRecordResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    review_diff_admin_records__record_id__review_diff_get: {
        parameters: {
            query?: {
                version?: number | null;
            };
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path: {
                record_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": {
                        [key: string]: unknown;
                    };
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    review_preview_admin_records__record_id__review_preview_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path: {
                record_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["PreviewRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": {
                        [key: string]: unknown;
                    };
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    transition_record_admin_records__record_id__transitions_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path: {
                record_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["TransitionRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["LifecycleRecordResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_record_versions_admin_records__record_id__versions_get: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path: {
                record_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["RecordHistoryResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_release_status_admin_releases_get: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ReleaseStatusResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    reload_table_admin_reload_table_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["ReloadTableRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ReloadTableResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    rollback_release_admin_rollback_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["RollbackRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["RollbackResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_source_documents_admin_source_documents_get: {
        parameters: {
            query?: {
                /** @description also list entries that were retired */
                include_retired?: boolean;
            };
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SourceDocumentListView"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    backfill_source_documents_admin_source_documents_backfill_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["BackfillView"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_blind_citations_admin_source_documents_blind_citations_get: {
        parameters: {
            query?: {
                /** @description which record versions to scan; 'published' is the criterion's population, 'all' measures every stored version */
                scope?: "published" | "all";
            };
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["BlindCitationListView"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_source_document_content_admin_source_documents_content_get: {
        parameters: {
            query: {
                /** @description the registered document id; the file is served byte-for-byte */
                document_id: string;
            };
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    ingest_from_url_admin_source_documents_from_url_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["UrlIngestRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["UrlIngestReport"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    register_source_document_admin_source_documents_register_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["RegistrationRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SourceDocumentView"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    check_registered_admin_source_documents_registered_get: {
        parameters: {
            query: {
                /** @description the citation, verbatim */
                document_id: string;
            };
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["RegisteredCheckView"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    retire_source_document_admin_source_documents_retire_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["RetireRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SourceDocumentView"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    update_source_document_admin_source_documents_update_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["RegistrationRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SourceDocumentView"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    upload_source_document_admin_source_documents_upload_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "multipart/form-data": components["schemas"]["Body_upload_source_document_admin_source_documents_upload_post"];
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["UploadReport"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    list_unanswered_admin_unanswered__get: {
        parameters: {
            query?: {
                /** @description Include resolved items too */
                resolved?: boolean | null;
            };
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["UnansweredList"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    warmup_admin_warmup_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": {
                        [key: string]: unknown;
                    };
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_meta_api_meta_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["MetaResponse"];
                };
            };
        };
    };
    preview_chunks_debug_preview_chunks_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["ChunkPreviewRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ChunkPreviewResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    health_check_health_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HealthResponse"];
                };
            };
        };
    };
    metrics_endpoint_metrics_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": {
                        [key: string]: unknown;
                    };
                };
            };
        };
    };
    query_query_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["SimpleQueryRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["QueryResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    compare_v2_compare_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["CompareRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ComparisonResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    feedback_v2_feedback_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["FeedbackRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["FeedbackResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    graph_stats_v2_graph_stats_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
        };
    };
    incremental_graph_update_v2_graph_update_post: {
        parameters: {
            query?: never;
            header?: {
                "X-Admin-Token"?: string | null;
            };
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": {
                    [key: string]: unknown;
                };
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    unified_query_v2_query_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["UnifiedQueryRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["UnifiedQueryResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    unified_query_stream_v2_query_stream_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["UnifiedQueryRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    post_recommend_v2_recommend_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["RecommendRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["RecommendResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    post_whatif_v2_whatif_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["WhatIfRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["WhatIfResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    v3_query_v3_query_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["V3QueryRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["V3QueryResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    warmup_endpoint_warmup_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": {
                        [key: string]: unknown;
                    };
                };
            };
        };
    };
}
// ---------------------------------------------------------------------------
// Hand-maintained compatibility layer (appended by scripts/gen-api-types.mjs).
// Derived from the generated schema types above - do not edit the field lists;
// edit the backend model and re-run `npm run types:gen`.
// ---------------------------------------------------------------------------

/** Enriched source object returned by /v2/query (schema: SourceChip). */
export type SourceChip = components["schemas"]["SourceChip"];

/** Router metadata returned by /v2/query (schema: UnifiedQueryResponse.router_info). */
export type RouterInfo = NonNullable<
  components["schemas"]["UnifiedQueryResponse"]["router_info"]
>;

/** Graph statistics returned by /v2/query (schema: UnifiedQueryResponse.graph_stats). */
export type GraphStats = NonNullable<
  components["schemas"]["UnifiedQueryResponse"]["graph_stats"]
>;
/** Per-claim provenance returned by /v2/query (schema: ClaimCitation, ticket 10). */
export type ClaimCitation = components["schemas"]["ClaimCitation"];
