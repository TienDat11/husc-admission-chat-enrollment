#!/usr/bin/env node
/* Generate FE API types from the BE OpenAPI schema — but degrade gracefully.
 *
 * The schema at ../api/openapi.json is an OPTIONAL build input: it only exists
 * when someone exports it from the running FastAPI backend. The committed
 * src/lib/api-types.ts is the source of truth otherwise. Before this guard,
 * a missing schema crashed `npm run dev` with a scary (but harmless) stack
 * trace via the predev hook. Now we skip cleanly when the schema is absent,
 * and only regenerate when it is present. Cross-platform, no shell builtins.
 */
import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const schema = resolve(here, "..", "..", "api", "openapi.json");
const out = resolve(here, "..", "src", "lib", "api-types.ts");

if (!existsSync(schema)) {
  console.log(
    `[types:gen] skip — OpenAPI schema not found at ${schema}. ` +
      `Using committed src/lib/api-types.ts. ` +
      `To regenerate: export the BE schema to that path, then re-run.`,
  );
  process.exit(0);
}

const res = spawnSync(
  process.platform === "win32" ? "npx.cmd" : "npx",
  ["openapi-typescript", schema, "-o", out],
  { stdio: "inherit" },
);
process.exit(res.status ?? 0);
