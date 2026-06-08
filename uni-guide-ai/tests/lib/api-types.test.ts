/**
 * Smoke test cho api-types.ts — verify file exists + is non-empty after codegen.
 * @spec(S10.A3)
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('api-types.ts codegen artifact', () => {
  it('file exists', () => {
    const p = resolve(__dirname, '../../src/lib/api-types.ts');
    const content = readFileSync(p, 'utf-8');
    expect(content.length).toBeGreaterThan(0);
  });

  it('is either placeholder OR has at least one paths/components export', () => {
    const p = resolve(__dirname, '../../src/lib/api-types.ts');
    const content = readFileSync(p, 'utf-8');
    const isPlaceholder = content.includes('_PlaceholderEmptySchema');
    const hasGenerated = /export\s+(type|interface)\s+(paths|components|operations)/.test(content);
    expect(isPlaceholder || hasGenerated).toBe(true);
  });
});
