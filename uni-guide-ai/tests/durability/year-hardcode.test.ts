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

  it('INTENDED-RED — durability risk: suggestedQuestions in chat-types.ts hardcodes 2024', () => {
    const chatTypesOccurrences = occurrences.filter(
      (o) => o.file.includes('chat-types') && o.year === '2024'
    );
    // This SHOULD fail if hardcode exists — proves the risk
    // chat-types.ts:57: "Điểm chuẩn ngành Công nghệ thông tin năm 2024?"
    expect(chatTypesOccurrences).toHaveLength(0); // INTENDED-RED: will fail while hardcode exists
  });

  it('INTENDED-RED — durability risk: suggestedQuestions in chat-types.ts hardcodes 2025-2026', () => {
    const chatTypesOccurrences = occurrences.filter(
      (o) => o.file.includes('chat-types') && (o.year === '2025' || o.year === '2026')
    );
    // chat-types.ts:58: "Học phí năm học 2025-2026 là bao nhiêu?"
    expect(chatTypesOccurrences).toHaveLength(0); // INTENDED-RED: will fail while hardcode exists
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
