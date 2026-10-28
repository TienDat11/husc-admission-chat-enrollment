/**
 * Ticket 23 criterion 3 — WCAG 2.2 AA contrast for the chat surface.
 *
 * Parses the REAL HSL tokens out of src/index.css (light :root block and the
 * .dark block) and computes the ratios for the pairs the chat components
 * actually use. The test therefore fails when someone lowers a token below
 * AA, not when a class name changes — it measures the shipped palette.
 *
 * Thresholds: 4.5:1 body text, 3:1 for large text / UI state borders and the
 * focus indicator (WCAG 1.4.11).
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const cssPath = resolve(__dirname, '../../src/index.css');
const css = readFileSync(cssPath, 'utf-8');

type Hsl = [number, number, number];
type Rgb = [number, number, number];

function blockOf(selector: string): string {
  const start = css.indexOf(selector);
  if (start === -1) throw new Error(`selector not found in index.css: ${selector}`);
  const open = css.indexOf('{', start);
  let depth = 0;
  for (let i = open; i < css.length; i += 1) {
    if (css[i] === '{') depth += 1;
    else if (css[i] === '}') {
      depth -= 1;
      if (depth === 0) return css.slice(open, i + 1);
    }
  }
  throw new Error(`unbalanced block for selector: ${selector}`);
}

function parseTokens(block: string): Record<string, Hsl> {
  const re = /--([a-z0-9-]+):\s*([\d.]+)\s+([\d.]+)%\s+([\d.]+)%\s*;/g;
  const out: Record<string, Hsl> = {};
  let m: RegExpExecArray | null;
  while ((m = re.exec(block)) !== null) {
    out[m[1]] = [Number(m[2]), Number(m[3]), Number(m[4])];
  }
  return out;
}

// Non-obvious conversions kept named: WCAG 2.x relative luminance pipeline.
function hslToRgb([h, s, l]: Hsl): Rgb {
  const sat = s / 100;
  const light = l / 100;
  const k = (n: number) => (n + h / 30) % 12;
  const a = sat * Math.min(light, 1 - light);
  const f = (n: number) =>
    light - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return [f(0), f(8), f(4)];
}

function luminance([r, g, b]: Rgb): number {
  const chan = (c: number) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  return 0.2126 * chan(r) + 0.7152 * chan(g) + 0.0722 * chan(b);
}

function contrast(a: Hsl, b: Hsl): number {
  const [l1, l2] = [luminance(hslToRgb(a)), luminance(hslToRgb(b))].sort((x, y) => y - x);
  return (l1 + 0.05) / (l2 + 0.05);
}

const WHITE: Hsl = [0, 0, 100]; // --chat-user-foreground / --primary-foreground

const light = parseTokens(blockOf(':root'));
const dark = parseTokens(blockOf('.dark'));

// Shared by every pair below; the message names the failing pair and ratio.
function expectContrast(label: string, actual: number, min: number): void {
  expect(actual, `${label} measured ${actual.toFixed(2)}:1 (needs ${min}:1)`).toBeGreaterThanOrEqual(min);
}

function crossCheck(theme: 'light' | 'dark', tokens: Record<string, Hsl>): void {
  const token = (name: string): Hsl => {
    const value = tokens[name];
    expect(value, `token --${name} must be defined in the ${theme} block`).toBeDefined();
    return value as Hsl;
  };

  const bot = token('chat-bot');

  // The answer text itself, plus the sidebar/input secondary text that sits
  // on --card / --background.
  expectContrast(`${theme}: --chat-bot-foreground on --chat-bot`, contrast(token('chat-bot-foreground'), bot), 4.5);
  expectContrast(`${theme}: --muted-foreground on --card`, contrast(token('muted-foreground'), token('card')), 4.5);
  expectContrast(`${theme}: --muted-foreground on --background`, contrast(token('muted-foreground'), token('background')), 4.5);

  // White text on the user bubble gradient endpoints.
  expectContrast(`${theme}: --chat-user-foreground on --chat-user-from`, contrast(WHITE, token('chat-user-from')), 4.5);
  expectContrast(`${theme}: --chat-user-foreground on --chat-user-to`, contrast(WHITE, token('chat-user-to')), 4.5);

  // Text tokens that sit on the bot bubble face.
  expectContrast(`${theme}: --chat-link on --chat-bot`, contrast(token('chat-link'), bot), 4.5);
  expectContrast(`${theme}: --chat-strong on --chat-bot`, contrast(token('chat-strong'), bot), 4.5);
  expectContrast(`${theme}: --chat-muted on --chat-bot`, contrast(token('chat-muted'), bot), 4.5);

  // Focus indicator + UI state borders need 3:1 (WCAG 1.4.11).
  expectContrast(`${theme}: --chat-focus-ring on --chat-bot`, contrast(token('chat-focus-ring'), bot), 3);
  expectContrast(`${theme}: --chat-focus-ring on --background`, contrast(token('chat-focus-ring'), token('background')), 3);
}

describe('ticket 23 criterion 3 — chat token contrast (WCAG 2.2 AA)', () => {
  it('meets AA in the light theme', () => {
    crossCheck('light', light);
  });

  it('meets AA in the dark theme', () => {
    crossCheck('dark', dark);
  });

  it('documents why the bubble and sidebar use dedicated tokens: white on brand colours is below AA', () => {
    // The pre-fix bubble gradient started at --primary, and the sidebar's
    // active row used --primary -> --secondary. This pins both defects so
    // nobody "simplifies" the tokens back to the brand pair.
    expect(contrast(WHITE, light.primary)).toBeLessThan(4.5);
    expect(contrast(WHITE, dark.secondary)).toBeLessThan(4.5);
    expect(contrast(WHITE, light['chat-user-from'])).toBeGreaterThanOrEqual(4.5);
    expect(contrast(WHITE, dark['chat-user-from'])).toBeGreaterThanOrEqual(4.5);
  });
});
