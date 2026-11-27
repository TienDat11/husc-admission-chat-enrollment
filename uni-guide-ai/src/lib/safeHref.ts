const SAFE_PROTOCOLS = new Set(["http:", "https:", "mailto:", "tel:"]);

/** Same-origin absolute path. Rejects protocol-relative "//host" and "/\\host" forms. */
const SAME_ORIGIN_PATH = /^\/(?![/\\])/;

/**
 * Returns the href when it is safe to render as a link, otherwise undefined.
 * Allows http(s), mailto, tel and same-origin paths; rejects javascript:, data:, etc.
 */
export function safeHref(href: string | null | undefined): string | undefined {
  if (typeof href !== "string") return undefined;
  const value = href.trim();
  if (!value) return undefined;
  if (SAME_ORIGIN_PATH.test(value)) return value;
  try {
    const url = new URL(value);
    return SAFE_PROTOCOLS.has(url.protocol) ? value : undefined;
  } catch {
    return undefined;
  }
}
