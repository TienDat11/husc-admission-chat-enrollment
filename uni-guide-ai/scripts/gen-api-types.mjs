#!/usr/bin/env node
/* Generate FE API types from the BE OpenAPI schema.
 *
 * The schema at ../api/openapi.json is a REQUIRED build input: it is committed
 * to the repo (export it with `python rag2025/scripts/export_openapi.py`), and
 * src/lib/api-types.ts is a generated artifact derived from it. A missing
 * schema used to make this script `process.exit(0)` quietly, which turned the
 * CI drift gate into a no-op: nothing regenerated, nothing compared, and the
 * FE shipped stale types that no test could catch. Now a missing schema is a
 * hard error.
 *
 * Hand-authored compatibility types (SourceChip / RouterInfo / GraphStats) are
 * appended after generation, derived from the generated schema types rather
 * than re-typed by hand, so `src/lib/api.ts` and `src/lib/chat-helpers.ts`
 * keep importing them from './api-types' and the drift gate still compares a
 * deterministic file.
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, resolve } from "node:path";
import openapiTs, { astToString } from "openapi-typescript";

const here = dirname(fileURLToPath(import.meta.url));
const schema = resolve(here, "..", "..", "api", "openapi.json");
const out = resolve(here, "..", "src", "lib", "api-types.ts");

if (!existsSync(schema)) {
  console.error(
    `[types:gen] FAIL: OpenAPI schema not found at ${schema}.\n` +
      "api-types.ts is generated FROM that file; without it the FE types are\n" +
      "unverifiable and the drift gate cannot run. Export it first:\n" +
      "  python rag2025/scripts/export_openapi.py",
  );
  process.exit(1);
}

const source = readFileSync(schema, "utf-8");

// Every type the compatibility block below derives must exist in the schema,
// or generation would emit a TS file that typechecks against nothing.
const REQUIRED_SCHEMA_TYPES = [
  ["SourceChip", ["components", "schemas", "SourceChip"]],
  ["UnifiedQueryResponse", ["components", "schemas", "UnifiedQueryResponse"]],
  ["ClaimCitation", ["components", "schemas", "ClaimCitation"]],
];

let parsed;
try {
  parsed = JSON.parse(source);
} catch (err) {
  console.error(`[types:gen] FAIL: ${schema} is not valid JSON: ${err.message}`);
  process.exit(1);
}

const missing = REQUIRED_SCHEMA_TYPES.filter(([, path]) => {
  let node = parsed;
  for (const key of path) {
    if (node == null || typeof node !== "object" || !(key in node)) return true;
    node = node[key];
  }
  return false;
}).map(([name]) => name);

if (missing.length) {
  console.error(
    `[types:gen] FAIL: ${schema} is missing schema type(s) the FE depends on: ` +
      `${missing.join(", ")}. Regenerate the export from the current backend.`,
  );
  process.exit(1);
}

// openapi-typescript v7 returns a TypeScript AST, not a string.
const generated = astToString(
  await openapiTs(pathToFileURL(schema), { exportType: true }),
);

// The generated file declares its types under components.schemas[...]. The FE
// imports three named types from this module; derive them from the schema
// instead of hand-copying fields, so a backend rename breaks generation or
// typecheck rather than silently shipping a stale shape.
const compat = [
  "// ---------------------------------------------------------------------------",
  "// Hand-maintained compatibility layer (appended by scripts/gen-api-types.mjs).",
  "// Derived from the generated schema types above - do not edit the field lists;",
  "// edit the backend model and re-run `npm run types:gen`.",
  "// ---------------------------------------------------------------------------",
  "",
  "/** Enriched source object returned by /v2/query (schema: SourceChip). */",
  'export type SourceChip = components["schemas"]["SourceChip"];',
  "",
  "/** Router metadata returned by /v2/query (schema: UnifiedQueryResponse.router_info). */",
  "export type RouterInfo = NonNullable<",
  '  components["schemas"]["UnifiedQueryResponse"]["router_info"]',
  ">;",
  "",
  "/** Graph statistics returned by /v2/query (schema: UnifiedQueryResponse.graph_stats). */",
  "export type GraphStats = NonNullable<",
  '  components["schemas"]["UnifiedQueryResponse"]["graph_stats"]',
  ">;",

  "/** Per-claim provenance returned by /v2/query (schema: ClaimCitation, ticket 10). */",
  'export type ClaimCitation = components["schemas"]["ClaimCitation"];',
  "",
].join("\n");

writeFileSync(out, generated.replace(/\s*$/, "\n") + compat, "utf-8");
console.log(`[types:gen] wrote ${out}`);
