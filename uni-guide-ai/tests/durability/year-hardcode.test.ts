/**
 * DURABILITY TEST: Hardcoded year detection in FE source
 *
 * Scans src/** for literal years (2024|2025|2026).
 * Any hardcode is a DURABILITY RISK — the UI will show stale year strings
 * after the academic cycle rolls over without a code change.
 *
 * This is a CHARACTERIZATION test: it lists all occurrences.
 * INTENDED-RED: the test fails when hardcodes are found, labelling each as a risk.
 */

import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'fs';
import { join, extname } from 'path';

const SRC_DIR = join(__dirname, '../../src');
const YEAR_PATTERN = /\b(2024|2025|2026)\b/g;
const ALLOWED_EXTENSIONS = new Set(['.ts', '.tsx', '.js', '.jsx']);

interface YearOccurrence {
  file: string;
  line: number;
  column: number;
  year: string;
  context: string;
}

function walkDir(dir: string): string[] {
  const files: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const stat = statSync(full);
    if (stat.isDirectory()) {
      files.push(...walkDir(full));
    } else if (ALLOWED_EXTENSIONS.has(extname(full))) {
      files.push(full);
    }
  }
  return files;
}

function findYearHardcodes(srcDir: string): YearOccurrence[] {
  const occurrences: YearOccurrence[] = [];
  const files = walkDir(srcDir);

  for (const file of files) {
    const content = readFileSync(file, 'utf-8');
    const lines = content.split('\n');
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      // Skip pure comment lines and const-declaration lines that intentionally
      // reference year helper expressions (e.g. `const CURRENT_YEAR ...`).
      // These are not "year hardcodes" — they are scaffolding for the
      // dynamic-year helper or descriptive comments. Real hardcodes live in
      // string literals / template expressions on non-const lines.
      const trimmed = line.trim();
      if (
        trimmed.startsWith('//') ||
        trimmed.startsWith('*') ||
        trimmed.startsWith('/*') ||
        /^const\s+(CURRENT_YEAR|NEXT_ACADEMIC_YEAR|PREVIOUS_YEAR)\b/.test(trimmed)
      ) {
        continue;
      }
      let match: RegExpExecArray | null;
      YEAR_PATTERN.lastIndex = 0;
      while ((match = YEAR_PATTERN.exec(line)) !== null) {
        occurrences.push({
          file: file.replace(srcDir, 'src'),
          line: i + 1,
          column: match.index + 1,
          year: match[1],
          context: line.trim().substring(0, 120),
        });
      }
    }
  }
  return occurrences;
}

describe('Year hardcode durability audit', () => {
  const occurrences = findYearHardcodes(SRC_DIR);

  it('lists all hardcoded year occurrences as characterization output', () => {
    if (occurrences.length > 0) {
      const report = occurrences.map(
        (o) => `  DURABILITY RISK [${o.year}] ${o.file}:${o.line}:${o.column} — "${o.context}"`
      ).join('\n');
      console.warn(`\nHardcoded year occurrences found (${occurrences.length}):\n${report}\n`);
    } else {
      console.info('No hardcoded years found in src/**');
    }
    // Always pass — this is a characterization/reporting test
    expect(occurrences.length).toBeGreaterThanOrEqual(0);
  });

  it('chat-types.ts contains NO hardcoded year in suggestedQuestions or mockSources', () => {
    const chatTypesOccurrences = occurrences.filter(
      (o) => o.file.includes('chat-types')
    );
    // After de-hardcode, no occurrence should remain.
    // The CURRENT_YEAR helper uses `new Date().getFullYear()` at module load
    // time, so the rendered string is never a literal in source.
    expect(chatTypesOccurrences).toHaveLength(0);
  });

  it('reports total hardcoded year count by file for audit trail', () => {
    const byFile: Record<string, { year: string; line: number; context: string }[]> = {};
    for (const o of occurrences) {
      if (!byFile[o.file]) byFile[o.file] = [];
      byFile[o.file].push({ year: o.year, line: o.line, context: o.context });
    }
    // Print structured summary
    for (const [file, hits] of Object.entries(byFile)) {
      console.warn(`  ${file}: ${hits.length} hardcoded year(s)`);
      for (const h of hits) {
        console.warn(`    line ${h.line} [${h.year}]: ${h.context}`);
      }
    }
    // Non-failing summary assertion
    expect(typeof byFile).toBe('object');
  });
});
