// Ticket T09 — "Script contrast/target trên build (như C1) = 0 vi phạm U1-U3".
// Serves uni-guide-ai/dist/, opens each /quan-ly route in headless Chrome,
// seeds the admin credential, stubs the admin APIs so each screen renders
// its real (anonymous-credential) content, evaluates checkPage
// (ported from mockups/check.mjs) per screen, and asserts zero U1-U3.
import http from "node:http";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DIST = path.resolve(__dirname, "../dist");

if (!existsSync(DIST)) {
  console.error("dist/ not found. Run `npm run build` first.");
  process.exit(1);
}

// SPA static server. Asset paths (/assets/*, /favicon...) are served from
// dist/; every other path is index.html with 200 at the original URL, so
// BrowserRouter keeps its location.
const MIME = {
  ".js": "application/javascript",
  ".css": "text/css",
  ".html": "text/html",
  ".json": "application/json",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".woff2": "font/woff2",
};
const INDEX = readFileSync(path.join(DIST, "index.html"));

const server = http.createServer((req, res) => {
  const reqPath = new URL(req.url, "http://localhost").pathname;
  const asset = path.join(DIST, reqPath);
  if (reqPath.startsWith("/assets/") || path.extname(reqPath) !== "") {
    if (existsSync(asset)) {
      res.writeHead(200, { "Content-Type": MIME[path.extname(asset)] ?? "application/octet-stream" });
      res.end(readFileSync(asset));
    } else {
      res.writeHead(404);
      res.end();
    }
    return;
  }
  res.writeHead(200, { "Content-Type": "text/html" });
  res.end(INDEX);
});

await new Promise((r) => server.listen(0, r));
const PORT = server.address().port;
const BASE = `http://127.0.0.1:${PORT}`;

// checkPage ported from mockups/check.mjs (pure, no deps).
const CHECK_PAGE = `(function(minTarget) {
  const BANNED = ["record","chunk","ingest","embedding","vector","token","LLM","upload","preview","publish","rollback","index","query","draft"];
  const parse = (c) => { const m = c.match(/rgba?\\(([^)]+)\\)/); if (!m) return null; const p = m[1].split(/[ ,/]+/).filter(Boolean).map(Number); return { r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1 }; };
  const lum = ({ r, g, b }) => { const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; }; return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b); };
  const ratio = (x, y) => { const a = lum(x), b = lum(y); return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05); };
  const bgOf = (el) => { for (let e = el; e; e = e.parentElement) { const c = parse(getComputedStyle(e).backgroundColor); if (c && c.a > 0.5) return c; } return { r: 255, g: 255, b: 255, a: 1 }; };
  const clipped = (s) => {
    const clip = s.clip || s.clipPath;
    if (!clip || clip === "none") return false;
    if (/rect\\(\\s*0(px)?[,\\s]+0(px)?[,\\s]+0(px)?[,\\s]+0(px)?\\s*\\)/.test(clip)) return true;
    if (clip === "inset(50%)") return true;
    return false;
  };
  const visible = (el) => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el); return r.width > 0 && r.height > 0 && s.visibility !== "hidden" && s.display !== "none" && r.right > 0 && r.left < innerWidth + 1 && !clipped(s); };

  const contrastFails = []; let textChecked = 0;
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const seen = new Set();
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    const el = n.parentElement; if (!n.textContent.trim() || seen.has(el) || !visible(el)) continue; seen.add(el);
    const s = getComputedStyle(el); const fg = parse(s.color); const bg = bgOf(el);
    const size = parseFloat(s.fontSize); const bold = parseInt(s.fontWeight, 10) >= 700;
    const large = size >= 24 || (bold && size >= 18.66);
    const need = large ? 3 : 4.5; const r = ratio(fg, bg); textChecked++;
    if (r < need && !el.closest("[disabled]")) contrastFails.push({ text: n.textContent.trim().slice(0, 40), ratio: +r.toFixed(2), need });
  }
  for (const el of document.querySelectorAll("input:not([type=checkbox]):not([type=radio]), select, textarea, .btn")) {
    if (!visible(el) || el.disabled) continue; textChecked++;
    const s = getComputedStyle(el); const bc = parse(s.borderTopColor); const r = ratio(bc, bgOf(el.parentElement));
    const fill = parse(s.backgroundColor); const r2 = fill && fill.a > 0.5 ? ratio(fill, bgOf(el.parentElement)) : 0;
    if (Math.max(r, r2) < 3) contrastFails.push({ text: "[vien] " + (el.textContent.trim().slice(0, 30) || el.id), ratio: +r.toFixed(2), need: 3 });
  }

  const targetFails = []; let targets = 0;
  for (const el of document.querySelectorAll("a, button, input, select, textarea, [role=tab]")) {
    if (!visible(el) || el.classList.contains("skip")) continue;
    if (el.tagName === "A" && el.closest("p, li") && !el.classList.contains("btn") && !el.classList.contains("cite") && !el.closest(".side")) continue;
    let box = el.getBoundingClientRect();
    if (el.type === "checkbox" || el.type === "radio") { const lab = el.closest("label"); if (lab) box = lab.getBoundingClientRect(); }
    targets++;
    if (box.width < minTarget - 0.5 || box.height < minTarget - 0.5) targetFails.push({ el: (el.textContent.trim() || el.id || el.type).slice(0, 40), w: Math.round(box.width), h: Math.round(box.height) });
  }

  const reflow = { scrollWidth: document.documentElement.scrollWidth, viewport: innerWidth };
  reflow.ok = reflow.scrollWidth <= innerWidth + 1;
  const text = document.body.innerText;
  const bannedHits = BANNED.filter((w) => new RegExp("\\\\b" + w + "\\\\b", "i").test(text));
  return { contrastFails, targetFails, reflow, bannedHits };
})()`;

// Admin API fixtures so the gate resolves and each screen shows real
// content. `/admin/index-jobs` (inside FIXTURE below) mirrors the job
// shape the release screen renders — a running job, a saved-but-
// unserved build, a failed job with retries — so every badge and row
// gets measured.
const IDENTITY = {
  actor_id: "cb.duyet",
  role: "reviewer_publisher",
  actions: [
    "records.read", "records.create", "records.edit", "records.submit", "records.approve", "records.reject",
    "records.publish", "records.rollback",
  ],
};

const RECORD = {
  record_id: "HOC_PHI:2026", record_type: "HOC_PHI", version: 2,
  admission_year: 2026, effective_from: "2026-01-01", effective_to: "2026-12-31",
  status: "review", authority_level: "husc_official", source_document_id: "hoc-phi-2026.pdf",
  source_url: null, source_page: null, published_at: null, authored_by: "cb.soan",
  supersedes: null, locale: "vi", text: "Học phí Hán Nôm 15tr.", values: {},
};

const FIXTURE = (urlPath) => {
  if (urlPath.startsWith("/admin/identity"))
    return { status: 200, body: IDENTITY };
  // fetchReviewDiff/fetchReviewPreview hit /admin/records/{id}/review-diff
  // (the id is URL-encoded), so match by suffix, not by the bare prefix.
  if (urlPath.includes("/review-diff"))
    return {
      status: 200,
      body: {
        record_id: "HOC_PHI:2026", draft_version: 2, high_impact_version: "2026.10",
        diff: { record_id: "HOC_PHI:2026", baseline_version: 1, draft_version: 2, high_impact_version: "2026.10", has_high_impact: false,
                changes: [{ field: "text", old: "14tr", new: "15tr", high_impact: false }] },
        validation_errors: [],
      },
    };
  if (urlPath.includes("/review-preview"))
    return {
      status: 200,
      body: {
        record_id: "HOC_PHI:2026", draft_version: 2, high_impact_version: "2026.10",
        diff: { record_id: "HOC_PHI:2026", baseline_version: 1, draft_version: 2, high_impact_version: "2026.10", has_high_impact: false, changes: [] },
        validation_errors: [],
        preview_status: "ok", answer: "Học phí Hán Nôm năm 2026 là 15 triệu.", chunks: [], sources: [],
      },
    };
  // GET /admin/records/{id} (fetchRecord) must answer with ONE
  // CanonicalRecord, not the list — RecordForm/ReviewPreview read
  // record fields directly off the response.
  if (/^\/admin\/records\/[^/]+/.test(urlPath))
    return { status: 200, body: RECORD };
  if (urlPath.startsWith("/admin/records"))
    return {
      status: 200,
      body: {
        total: 1,
        records: [RECORD],
      },
    };
  if (urlPath.startsWith("/admin/unanswered"))
    return {
      status: 200,
      body: {
        count: 1,
        items: [
          { question: "Trường có ký túc xá không?", trace_id: "t-1", asked_at: "2026-10-06T08:14:00Z",
            route: "padded_rag", status_code: "NO_INFO", resolved: false,
            resolved_at: null, resolved_by_record_id: null },
        ],
      },
    };
  if (urlPath.startsWith("/admin/audit"))
    return {
      status: 200,
      body: {
        count: 1,
        events: [
          { seq: 1, prev_hash: "a", entry_hash: "b",
            event: { action: "publish", actor_id: "cb.duyet", entity_id: "HOC_PHI:2026", reason: "đã duyệt", timestamp: "2026-10-05T09:00:00Z" } },
        ],
      },
    };
  if (urlPath.startsWith("/admin/releases"))
    return {
      status: 200,
      body: {
        active_release_id: "R-1", previous_release_id: "R-0",
        active: { release_id: "R-1", created_at: "2026-10-06T09:12:00Z" },
        previous: { release_id: "R-0", created_at: "2026-09-01T08:00:00Z" },
        problems: [],
      },
    };
  if (urlPath.startsWith("/admin/index-jobs"))
    return {
      status: 200,
      body: {
        reindex_needed: true,
        reindex_reason: "record count changed",
        pending_outbox_entries: 0,
        pending_affected_record_ids: [],
        problems: [],
        jobs: [
          { job_id: "idx-1", state: "running", label: "Đang chạy", saved: false, serving: false, retry: [] },
          { job_id: "idx-2", state: "failed", label: "Thất bại", saved: true, serving: false, reason: "lỗi mẫu", retry: [{ step: "embed", runnable: true, command: "python -m ingest", reason: "" }] },
          { job_id: "idx-3", state: null, label: "Trên disk", saved: true, serving: false, staged_dir: "/var/lib/husc/staging/2026", retry: [] },
        ],
      },
    };
  if (urlPath.startsWith("/v2/query"))
    return {
      status: 200,
      body: { answer: "Điểm chuẩn ngành Hán Nôm năm 2025 là 24,5.", status_code: "SUCCESS",
              sources: [{ id: "s1", title: "diem-chuan-2025.pdf", snippet: "Bảng điểm chuẩn", data_year: "2025" }],
              data_gap_hints: [] },
    };
  return { status: 404, body: {} };
};

const puppeteer = await import("puppeteer-core");
const browser = await puppeteer.default.launch({
  headless: "new",
  executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
  args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-gpu"],
});

// Each route: where to navigate, the wait target proving its own content
// rendered (not the login gate, not NotFound), and the heading. The
// /admin routes are covered alongside /quan-ly: ConfigEditor, the
// records CRUD (list, create, edit, review+preview), the release
// screen and the three ComingSoon placeholders.
const ROUTES = [
  { path: "/quan-ly", wait: "Việc cần làm hôm nay", seed: true },
  { path: "/quan-ly/them", wait: "Thêm tài liệu" },
  { path: "/quan-ly/them/qa", wait: "Gõ hỏi–đáp trực tiếp" },
  { path: "/quan-ly/cho-duyet", wait: "Chờ duyệt" },
  { path: "/quan-ly/kho", wait: "Kho thông tin" },
  { path: "/quan-ly/cau-hoi", wait: "Câu hỏi bot chưa trả lời" },
  { path: "/quan-ly/hoi-thu", wait: "Hỏi thử bot" },
  { path: "/quan-ly/lich-su", wait: "Lịch sử & khôi phục" },
  { path: "/admin", wait: "Config" },
  { path: "/admin/config", wait: "Config" },
  { path: "/admin/records", wait: "Mã record" },
  { path: "/admin/records/new", wait: "Mã record" },
  { path: "/admin/records/HOC_PHI:2026/edit", wait: "Mã record" },
  { path: "/admin/records/HOC_PHI:2026/review", wait: "Xem trước trước khi gửi duyệt" },
  { path: "/admin/releases", wait: "Release & index job" },
  { path: "/admin/ingest", wait: "Ingest" },
  { path: "/admin/retrieval", wait: "Retrieval" },
  { path: "/admin/debt", wait: "Debt" },
];

// Both themes: the office shell is light by contract (05-ux-spec ships one
// token set), so a user whose OS prefers dark must still get a legible shell.
// Running dark catches a `dark:` utility leaking into the shell or a token
// utility resolving to the dark palette on a hardcoded white surface.
const THEMES = ["light", "dark"];

let totalContrast = 0;
let totalTarget = 0;
let totalBanned = 0;
let totalReflow = 0;
let failed = 0;

for (const theme of THEMES) {
for (const route of ROUTES) {
  const page = await browser.newPage();
  await page.setViewport({ width: 320, height: 812, deviceScaleFactor: 1 });

  // Seed the credential before the SPA boots so the gate resolves to
  // authenticated (otherwise every screen renders the login form). Pin the
  // theme too: without it the SPA falls back to prefers-color-scheme, which is
  // light in headless Chrome and would silently skip the dark pass.
  await page.evaluateOnNewDocument((themeName) => {
    sessionStorage.setItem("husc.admin.token", "audit-token");
    localStorage.setItem("theme", themeName);
  }, theme);

  // Stub every admin API with fixtures so screens show real content. The SPA
  // calls VITE_API_BASE_URL (http://localhost:8000 by default), a different
  // origin from this static server, so match on pathname and answer the CORS
  // preflight — otherwise the browser blocks the stubbed response.
  const CORS_HEADERS = {
    "access-control-allow-origin": "*",
    "access-control-allow-headers": "*",
    "access-control-allow-methods": "GET,POST,PUT,PATCH,DELETE,OPTIONS",
  };
  await page.setRequestInterception(true);
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (
      request.method() === "OPTIONS" &&
      (url.pathname.startsWith("/admin/") || url.pathname.startsWith("/v2/"))
    ) {
      request.respond({ status: 204, headers: CORS_HEADERS, body: "" });
      return;
    }
    // Stub API calls only. A document request to /admin/<route> shares
    // the path prefix — answering it with fixture JSON would stop the
    // SPA from ever booting on that route (wait text never appears).
    const isApiCall =
      request.resourceType() === "xhr" || request.resourceType() === "fetch";
    if (
      isApiCall &&
      (url.pathname.startsWith("/admin/") || url.pathname.startsWith("/v2/"))
    ) {
      const fixture = FIXTURE(url.pathname + url.search);
      request.respond({
        status: fixture.status,
        contentType: "application/json",
        headers: CORS_HEADERS,
        body: JSON.stringify(fixture.body),
      });
      return;
    }
    request.continue();
  });

  await page.goto(`${BASE}${route.path}`, { waitUntil: "networkidle0", timeout: 30000 });
  await page.waitForFunction(
    (needle) => document.body.innerText.includes(needle),
    { timeout: 20000 },
    route.wait,
  );
  // Let react-query settle and fonts paint before measuring.
  await new Promise((r) => setTimeout(r, 800));

  // Target-size floor (WCAG 2.5.5 AAA = 44px; WCAG 2.5.8 AA = 24px).
  // The teacher/student shell (/quan-ly) is held to the 44px AAA floor it
  // was gated on. The /admin operator console is a dense technical surface
  // (h-7/h-8 controls by design) with no AAA requirement — 24px (AA) is
  // the honest floor there, matching the contrast pass's AA scope.
  const minTarget = route.path.startsWith("/quan-ly") ? 44 : 24;
  const result = await page.evaluate(CHECK_PAGE, minTarget);
  // The banned-word list guards student/teacher-facing copy (no RAG
  // jargon in the shell). /admin is the operator console: record,
  // ingest, embedding, publish are its actual vocabulary, so the
  // list only applies to /quan-ly.
  const bannedHits = route.path.startsWith("/quan-ly")
    ? result.bannedHits
    : [];
  totalContrast += result.contrastFails.length;
  totalTarget += result.targetFails.length;
  totalBanned += bannedHits.length;
  totalReflow += result.reflow.ok ? 0 : 1;

  const line = `[${theme}/${route.path}] contrast=${result.contrastFails.length} target=${result.targetFails.length} banned=${bannedHits.length} reflow=${result.reflow.ok ? "ok" : "FAIL"}`;
  if (result.contrastFails.length || result.targetFails.length || bannedHits.length || !result.reflow.ok) failed += 1;
  console.log(line);
  if (result.contrastFails.length) console.log("  contrast:", JSON.stringify(result.contrastFails, null, 2));
  if (result.targetFails.length) console.log("  target:", JSON.stringify(result.targetFails, null, 2));
  if (bannedHits.length) console.log("  banned:", bannedHits);
  if (!result.reflow.ok) console.log("  reflow:", JSON.stringify(result.reflow));

  await page.close();
}
}

await browser.close();
server.close();

const total = totalContrast + totalTarget + totalReflow;
console.log(`\nTotal U1-U3 violations: ${total}`);
console.log(`  contrast (U1/U2): ${totalContrast}`);
console.log(`  target (U3):      ${totalTarget}`);
console.log(`  reflow (U5):      ${totalReflow}`);
console.log(`  banned:           ${totalBanned}`);

if (failed > 0 || total > 0) {
  console.error("\nFAIL — U1-U3 violations found.");
  process.exit(1);
} else {
  console.log("\nPASS — 0 U1-U3 violations on build.");
}
