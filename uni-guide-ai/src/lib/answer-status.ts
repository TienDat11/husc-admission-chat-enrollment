/**
 * Ticket 17: answer-surface state classification.
 *
 * Every predicate here is keyed on fields the backend ACTUALLY ships on the
 * /v2 wire (measured, rag2025/src/services/llm_generator.py:873 and the
 * main.py meta frame):
 *
 * - `provider === "evidence-policy:<reason>"` — a policy-authored refusal or
 *   conflict replaced the model answer. Reasons (abstention_policy.py:60-65):
 *   `no-evidence`, `year-mismatch`, `lineage-missing`, `out-of-scope-major`
 *   (abstain) and `unresolved-conflict` (conflict). These arrive with
 *   `status_code: SUCCESS` and `answer_confidence: null`, so NOTHING ELSE on
 *   the payload distinguishes them from a normal answer — keying off
 *   provider is the only honest read of the wire.
 * - `status_code` in {INSUFFICIENT_DATA, HUSC_ENTITY_NOT_FOUND} — the
 *   guardrail pre-check abstention (provider null: the generator never ran).
 * - `source.data_year` — the effective year of each cited chunk
 *   (SourceChip.data_year, main.py:601).
 * - `currentAdmissionYear` — from the metadata API `/api/meta`
 *   (MetaResponse.current_admission_year, rag2025/src/routers/meta.py:37).
 *   NEVER hardcoded: the stale state must survive the academic-cycle
 *   rollover without a code change.
 *
 * A state that no real payload can produce is not a state; nothing here
 * invents one.
 */

/** Prefix the BE puts on `provider` for every policy-authored answer. */
export const EVIDENCE_POLICY_PREFIX = 'evidence-policy:';

/** The one policy reason that is a conflict, not an abstention. */
export const CONFLICT_REASON = 'unresolved-conflict';

/** True when the answer text was authored by abstention policy (not a model). */
export function isPolicyAbstention(provider?: string | null): boolean {
  return (
    typeof provider === 'string' &&
    provider.startsWith(EVIDENCE_POLICY_PREFIX) &&
    provider.slice(EVIDENCE_POLICY_PREFIX.length) !== CONFLICT_REASON
  );
}

/** True when the answer reports an unresolved multi-source conflict. */
export function isPolicyConflict(provider?: string | null): boolean {
  return provider === `${EVIDENCE_POLICY_PREFIX}${CONFLICT_REASON}`;
}

/**
 * Guardrail abstain status codes (rag2025 guardrail pre-check; these arrive
 * with provider null because the generator never runs). They decorate their
 * own warning banner via decorateAnswer, and the surface classifier must NOT
 * re-label them as 'stale' when their chip list happens to carry old years.
 */
export const ABSTAIN_STATUSES: readonly string[] = ['INSUFFICIENT_DATA', 'HUSC_ENTITY_NOT_FOUND'];

export function isAbstainStatus(statusCode?: string): boolean {
  return statusCode !== undefined && ABSTAIN_STATUSES.includes(statusCode);
}

/**
 * Pull a bare 4-digit year out of a chip's `data_year` (e.g. "20YY",
 * "20YY/20YY", "N/A", undefined). Null when unparseable — an unknown year
 * is never stale.
 */
export function parseSourceYear(dataYear?: string): number | null {
  if (!dataYear) return null;
  const m = dataYear.match(/\d{4}/);
  return m ? Number(m[0]) : null;
}

/**
 * Stale = the cited data predates the CURRENT admission year reported by the
 * metadata API. Equal or newer is not stale. Unknown on either side is not
 * stale (fail-open: never dress missing information as outdated information).
 */
export function isStaleSource(
  dataYear: string | undefined,
  currentAdmissionYear: number | null,
): boolean {
  if (currentAdmissionYear === null) return false;
  const year = parseSourceYear(dataYear);
  return year !== null && year < currentAdmissionYear;
}

/** True when at least one cited source is stale. */
export function hasStaleSources(
  sources: Array<{ data_year?: string }> | undefined,
  currentAdmissionYear: number | null,
): boolean {
  return (sources || []).some((s) => isStaleSource(s.data_year, currentAdmissionYear));
}

/**
 * The four mutually exclusive surface states ticket 17 requires to be
 * distinguishable. Precedence is deliberate:
 * 1. conflict  — a policy conflict outranks everything (never self-select).
 * 2. abstain   — a policy refusal (provider prefix) or a guardrail refusal
 *                (status code) outranks staleness: the answer is a refusal,
 *                so its "verified portion" years are context, not the
 *                answer's freshness.
 * 3. stale     — a normal answer whose citations predate the current year.
 * 4. normal    — everything else (transport errors are Message.errorKind,
 *                a different rendering path, not an answer state).
 */
export type AnswerSurfaceState = 'normal' | 'abstain' | 'conflict' | 'stale';

export function deriveSurfaceState(
  message: {
    provider?: string | null;
    statusCode?: string;
    sources?: Array<{ data_year?: string }>;
  },
  currentAdmissionYear: number | null,
): AnswerSurfaceState {
  if (isPolicyConflict(message.provider)) return 'conflict';
  if (isPolicyAbstention(message.provider) || isAbstainStatus(message.statusCode)) return 'abstain';
  // Stale is a judgment about a DELIVERED answer: a GENERATION_FAILED or
  // scope-blocked bubble is an error/banner state, not an old-data state.
  // (The hook defaults statusCode to 'SUCCESS' when the BE omits it, so
  // `undefined` only occurs on hand-built messages in tests.)
  const delivered = message.statusCode === undefined || message.statusCode === 'SUCCESS';
  if (delivered && hasStaleSources(message.sources, currentAdmissionYear)) return 'stale';
  return 'normal';
}
