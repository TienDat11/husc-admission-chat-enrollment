/**
 * Ticket 17: answer-surface state classifier (src/lib/answer-status.ts).
 *
 * The states here exist ONLY because the real /v2 payload can produce them
 * (measured 2026-09-13):
 * - abstain/conflict: llm_generator.py:873 sets provider to
 *   "evidence-policy:<reason>" with status_code SUCCESS — provider is the
 *   only field that separates a policy refusal from a model answer.
 * - stale: SourceChip.data_year vs /api/meta current_admission_year.
 *
 * Criterion 7 (stale distinguishable from abstain and conflict) is proven by
 * the precedence tests: a payload that is BOTH policy-authored AND carries
 * old-year chips resolves to the policy state, never to 'stale'.
 */
import { describe, it, expect } from 'vitest';
import {
  EVIDENCE_POLICY_PREFIX,
  CONFLICT_REASON,
  isPolicyAbstention,
  isPolicyConflict,
  isAbstainStatus,
  parseSourceYear,
  isStaleSource,
  hasStaleSources,
  deriveSurfaceState,
} from '@/lib/answer-status';

describe('isPolicyAbstention — provider-keyed, per the real wire', () => {
  // The four abstain reasons in abstention_policy.py:60-65.
  it.each(['no-evidence', 'year-mismatch', 'lineage-missing', 'out-of-scope-major'])(
    'abstain for reason "%s"',
    (reason) => {
      expect(isPolicyAbstention(`${EVIDENCE_POLICY_PREFIX}${reason}`)).toBe(true);
    },
  );

  it('is FALSE for the conflict reason (that is its own state)', () => {
    expect(isPolicyConflict(`${EVIDENCE_POLICY_PREFIX}${CONFLICT_REASON}`)).toBe(true);
    expect(isPolicyAbstention(`${EVIDENCE_POLICY_PREFIX}${CONFLICT_REASON}`)).toBe(false);
  });

  it('is FALSE for model providers and for null (Main: scope block sends provider null)', () => {
    expect(isPolicyAbstention('llama-3.3-70b')).toBe(false);
    // Season suffixes ride the model name, never the policy prefix.
    expect(isPolicyAbstention('Fallback(Error) + season-fallback')).toBe(false);
    expect(isPolicyAbstention(null)).toBe(false);
    expect(isPolicyAbstention(undefined)).toBe(false);
  });
});

describe('isStaleSource — data_year vs the metadata-API year', () => {
  it('stale when the cited year predates the current admission year', () => {
    expect(isStaleSource('2024', 2026)).toBe(true);
  });

  it('not stale when equal or newer', () => {
    expect(isStaleSource('2026', 2026)).toBe(false);
    expect(isStaleSource('2027', 2026)).toBe(false);
  });

  it('fail-open on unparseable year and on unknown current year', () => {
    expect(isStaleSource('N/A', 2026)).toBe(false);
    expect(isStaleSource(undefined, 2026)).toBe(false);
    expect(isStaleSource('2024', null)).toBe(false);
  });

  it('parses a year embedded in a range label', () => {
    expect(parseSourceYear('2025/2026')).toBe(2025);
    expect(isStaleSource('2025/2026', 2027)).toBe(true);
  });
});


describe('isAbstainStatus — guardrail refusals (provider null on the wire)', () => {
  it('recognizes both guardrail codes', () => {
    expect(isAbstainStatus('INSUFFICIENT_DATA')).toBe(true);
    expect(isAbstainStatus('HUSC_ENTITY_NOT_FOUND')).toBe(true);
  });

  it('is FALSE for SUCCESS, scope block, PII block, and generation failure', () => {
    expect(isAbstainStatus('SUCCESS')).toBe(false);
    expect(isAbstainStatus('NOT_IN_HUSC_SCOPE')).toBe(false);
    expect(isAbstainStatus('SENSITIVE_PII_DETECTED')).toBe(false);
    expect(isAbstainStatus('GENERATION_FAILED')).toBe(false);
    expect(isAbstainStatus(undefined)).toBe(false);
  });
});
describe('deriveSurfaceState — the four states are mutually exclusive (criterion 7)', () => {
  const staleChip = { data_year: '2024' };
  const currentChip = { data_year: '2026' };

  it('conflict outranks stale: policy conflict + old chips -> conflict', () => {
    expect(
      deriveSurfaceState(
        { provider: `${EVIDENCE_POLICY_PREFIX}${CONFLICT_REASON}`, sources: [staleChip] },
        2026,
      ),
    ).toBe('conflict');
  });

  it('abstain outranks stale: year-mismatch refusal cites old years BY DESIGN -> abstain', () => {
    expect(
      deriveSurfaceState(
        { provider: `${EVIDENCE_POLICY_PREFIX}year-mismatch`, sources: [staleChip] },
        2026,
      ),
    ).toBe('abstain');
  });

  it('stale only for non-policy answers with an old citation', () => {
    expect(deriveSurfaceState({ provider: 'llama-3.3-70b', sources: [staleChip] }, 2026)).toBe(
      'stale',
    );
    // provider null (no model, no policy) still reads its chips.
    expect(deriveSurfaceState({ provider: null, sources: [staleChip] }, 2026)).toBe('stale');
  });

  it('normal when current, empty, or the metadata year is unknown', () => {
    expect(deriveSurfaceState({ provider: 'llama-3.3-70b', sources: [currentChip] }, 2026)).toBe(
      'normal',
    );
    expect(deriveSurfaceState({ provider: 'llama-3.3-70b', sources: [] }, 2026)).toBe('normal');
    // Unknown current year => never claim stale (fail-open).
    expect(deriveSurfaceState({ provider: 'llama-3.3-70b', sources: [staleChip] }, null)).toBe(
      'normal',
    );
  });

  it('guardrail abstain (status code, provider null) is NOT re-labeled stale despite old chips', () => {
    expect(
      deriveSurfaceState(
        { provider: null, statusCode: 'INSUFFICIENT_DATA', sources: [{ data_year: '2019' }] },
        2026,
      ),
    ).toBe('abstain');
  });

  it('hasStaleSources ignores chips without a year', () => {
    expect(hasStaleSources([{ data_year: 'N/A' }, staleChip], 2026)).toBe(true);
    expect(hasStaleSources([{ data_year: 'N/A' }], 2026)).toBe(false);
    expect(hasStaleSources(undefined, 2026)).toBe(false);
  });
});
