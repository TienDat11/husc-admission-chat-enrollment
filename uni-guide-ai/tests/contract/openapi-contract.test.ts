/**
 * Cross-stack contract test (ticket 42).
 *
 * Reads the REAL committed schema at `api/openapi.json` (exported from the
 * FastAPI app by `python rag2025/scripts/export_openapi.py`) and compares it
 * against the shapes the FE client actually declares in
 * `src/lib/adminApi.ts` / `src/lib/api.ts` and the enum values in the Python
 * source. Nothing here mocks or re-states the contract: a backend rename or an
 * enum change makes THIS test red, which is precisely what the mocked
 * `tests/admin/records.test.tsx` can never do.
 *
 * The FE side is parsed with the TypeScript compiler API (a devDependency),
 * so the source of truth for "what the FE expects" is the FE's own
 * declarations, not a third hand-written copy.
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, URL as NodeURL } from 'node:url';
import { describe, expect, it } from 'vitest';
import type tsModule from 'typescript';
import type {
  LiteralTypeNode,
  PropertySignature,
  StringLiteral,
  UnionTypeNode,
} from 'typescript';

// jsdom replaces the global URL, whose output fileURLToPath rejects - use node's.
const here = fileURLToPath(new NodeURL('.', import.meta.url));
const repoRoot = resolve(here, '..', '..', '..');

// `typescript` is a devDependency of uni-guide-ai; import it via CJS require so
// this file works regardless of the package's ESM interop shape.
const require = createRequire(import.meta.url);
const ts: typeof tsModule = require('typescript');

type OpenApiSchema = {
  paths: Record<string, Record<string, { parameters?: unknown[]; responses?: Record<string, any> }>>;
  components: { schemas: Record<string, { properties?: Record<string, unknown>; enum?: string[] }> };
};

const schema: OpenApiSchema = JSON.parse(
  readFileSync(resolve(repoRoot, 'api', 'openapi.json'), 'utf-8'),
);

const adminApiSource = readFileSync(
  resolve(repoRoot, 'uni-guide-ai', 'src', 'lib', 'adminApi.ts'),
  'utf-8',
);
const apiSource = readFileSync(resolve(repoRoot, 'uni-guide-ai', 'src', 'lib', 'api.ts'), 'utf-8');

/** String-literal members of `export type NAME = "a" | "b" | ...;`. */
function unionMembers(source: string, name: string): string[] {
  const file = ts.createSourceFile('fe.ts', source, ts.ScriptTarget.Latest, true);
  for (const stmt of file.statements) {
    if (
      ts.isTypeAliasDeclaration(stmt) &&
      stmt.name.text === name &&
      stmt.type.kind === ts.SyntaxKind.UnionType
    ) {
      const members = (stmt.type as UnionTypeNode).types
        .filter((t) => t.kind === ts.SyntaxKind.LiteralType)
        .map((t) => (t as LiteralTypeNode).literal)
        .filter((l) => l.kind === ts.SyntaxKind.StringLiteral)
        .map((l) => (l as StringLiteral).text);
      if (members.length) return members;
    }
  }
  throw new Error(`FE union ${name} not found or has no string members`);
}

/** Declared property names of `export interface NAME { ... }`. */
function interfaceMembers(source: string, name: string): string[] {
  const file = ts.createSourceFile('fe.ts', source, ts.ScriptTarget.Latest, true);
  for (const stmt of file.statements) {
    if (ts.isInterfaceDeclaration(stmt) && stmt.name.text === name) {
      return stmt.members
        .filter((m) => ts.isPropertySignature(m) && m.name != null)
        .map((m) => {
          const n = (m as PropertySignature).name!;
          return ts.isIdentifier(n) ? n.text : (n as StringLiteral).text;
        });
    }
  }
  throw new Error(`FE interface ${name} not found`);
}

/** Value of `export const NAME = "string literal";` - the FE's own wording. */
function constSourceString(source: string, name: string): string {
  const file = ts.createSourceFile('fe.ts', source, ts.ScriptTarget.Latest, true);
  for (const stmt of file.statements) {
    if (!ts.isVariableStatement(stmt)) continue;
    for (const decl of stmt.declarationList.declarations) {
      if (!ts.isIdentifier(decl.name) || decl.name.text !== name) continue;
      const init = decl.initializer;
      if (init && ts.isStringLiteral(init)) return init.text;
    }
  }
  throw new Error(`FE string const ${name} not found in source`);
}

/** `NAME = "value"` members of a Python `class NAME(str, Enum):` block. */
function pythonEnumMembers(relPath: string, className: string): string[] {
  const src = readFileSync(resolve(repoRoot, relPath), 'utf-8');
  const lines = src.split(/\r?\n/);
  const start = lines.findIndex((l) => new RegExp(`^class ${className}\\b`).test(l));
  if (start === -1) throw new Error(`Python class ${className} not found in ${relPath}`);
  const members: string[] = [];
  for (let i = start + 1; i < lines.length; i++) {
    const line = lines[i];
    if (/^\S/.test(line)) break; // dedent = end of class body
    const m = line.match(/^\s+[A-Z][A-Z0-9_]*\s*=\s*"([^"]+)"/);
    if (m) members.push(m[1]);
  }
  if (!members.length) throw new Error(`no enum members parsed for ${className}`);
  return members;
}

function schemaProps(component: string): string[] {
  const entry = schema.components.schemas[component];
  if (!entry) throw new Error(`schema component ${component} missing from api/openapi.json`);
  return Object.keys(entry.properties ?? {});
}

function schemaEnum(component: string): string[] {
  const entry = schema.components.schemas[component];
  if (!entry) throw new Error(`schema component ${component} missing from api/openapi.json`);
  if (!Array.isArray(entry.enum)) {
    throw new Error(`schema component ${component} has no enum`);
  }
  return [...entry.enum];
}

function sorted(values: string[]): string[] {
  return [...values].sort();
}

const ADMIN_PATHS = [
  ['/admin/records', 'post'],
  ['/admin/records', 'get'],
  ['/admin/records/{record_id}/edits', 'post'],
  ['/admin/records/{record_id}/transitions', 'post'],
  ['/admin/identity', 'get'],
] as const;

describe('api/openapi.json is the committed contract artifact', () => {
  it('carries the admin endpoints the FE client calls', () => {
    for (const [path, method] of ADMIN_PATHS) {
      expect(schema.paths[path], `path ${path} missing from schema`).toBeDefined();
      expect(schema.paths[path][method], `${method.toUpperCase()} ${path} missing`).toBeDefined();
    }
  });

  it('declares the FE token header on every admin operation', () => {
    // The name comes from adminApi.ts's own constant, not a third literal here:
    // a FE-side rename that the backend has not adopted must go RED, and vice
    // versa the schema param is what the server actually reads.
    //
    // The param is optional on purpose (ticket T08): ``POST /admin/login``
    // issues an HttpOnly session cookie, and ``current_identity`` accepts
    // either credential — header first, cookie second. OpenAPI cannot spell
    // "header OR cookie", so the header is documented as optional and the
    // cookie path is the second alternative. What this test still catches is
    // the drift that matters: the header param disappearing from an admin
    // operation, or the FE renaming it without the backend following.
    const header = constSourceString(adminApiSource, 'ADMIN_TOKEN_HEADER');
    for (const [path, method] of ADMIN_PATHS) {
      const params = (schema.paths[path][method]?.parameters ?? []) as Array<{
        in: string;
        name: string;
        required?: boolean;
      }>;
      const token = params.find((p) => p.in === 'header' && p.name === header);
      expect(token, `${method.toUpperCase()} ${path} has no ${header} header param`).toBeDefined();
      expect(token!.required, `${header} must stay optional while the session cookie is accepted`).toBe(
        false,
      );
    }
  });

  it('maps the response schemas the FE types are named after', () => {
    const ref = (path: string, method: string) =>
      schema.paths[path][method]?.responses?.['200']?.content?.['application/json']?.schema?.$ref;
    expect(ref('/admin/records', 'post')).toBe('#/components/schemas/LifecycleRecordResponse');
    expect(ref('/admin/records/{record_id}/edits', 'post')).toBe(
      '#/components/schemas/LifecycleRecordResponse',
    );
    expect(ref('/admin/records/{record_id}/transitions', 'post')).toBe(
      '#/components/schemas/LifecycleRecordResponse',
    );
    expect(ref('/admin/records', 'get')).toBe('#/components/schemas/RecordListResponse');
    expect(ref('/admin/identity', 'get')).toBe('#/components/schemas/AdminIdentityResponse');
    expect(ref('/v2/query', 'post')).toBe('#/components/schemas/UnifiedQueryResponse');
  });
});

describe('enum parity: FE unions vs the schema / Python source', () => {
  it('LifecycleAction union equals the schema enum', () => {
    expect(sorted(unionMembers(adminApiSource, 'LifecycleAction'))).toEqual(
      sorted(schemaEnum('LifecycleAction')),
    );
  });

  it('RecordStatus union equals the schema enum', () => {
    expect(sorted(unionMembers(adminApiSource, 'RecordStatus'))).toEqual(
      sorted(schemaEnum('RecordStatus')),
    );
  });

  // JobState / AdminAction are backend-internal enums: their values reach the
  // wire as plain strings (IndexJobResponse.state, AdminIdentityResponse.actions),
  // so they never appear as a schema `enum`. Compare against the Python source
  // the schema is generated from - both sides of this check stay in the repo.
  it('JobStateName union equals the Python JobState enum', () => {
    expect(sorted(unionMembers(adminApiSource, 'JobStateName'))).toEqual(
      sorted(pythonEnumMembers('rag2025/src/services/reindex_job.py', 'JobState')),
    );
  });

  it('AdminActionName union is a subset of the Python AdminAction enum', () => {
    // FE deliberately wires only part of the grant matrix; the server owns the
    // rest. A name the FE sends that the server does not know is the drift here.
    const be = pythonEnumMembers('rag2025/src/domain/admin_identity.py', 'AdminAction');
    const fe = unionMembers(adminApiSource, 'AdminActionName');
    const unknown = fe.filter((name) => !be.includes(name));
    expect(unknown, `FE sends AdminAction names the backend does not define`).toEqual([]);
  });
});

describe('field parity: FE interfaces vs schema properties', () => {
  // Every field the FE declares must exist in the schema: a backend rename or
  // removal makes the FE declare a field the schema no longer has, and THIS
  // test, not a human, notices.
  //
  // The reverse direction is only asserted where the schema component is
  // closed. CanonicalRecord declares additionalProperties: true - the backend
  // adds record fields by design - so there a schema-only field is legal and
  // a strict equality would fire false REDs on every additive BE change.
  const PAIRS: Array<[
    label: string,
    feNames: string[],
    schemaComponent: string,
    exactBothWays: boolean,
  ]> = [
    ['CanonicalRecord', interfaceMembers(adminApiSource, 'CanonicalRecord'), 'CanonicalRecord', false],
    ['AdminIdentity', interfaceMembers(adminApiSource, 'AdminIdentity'), 'AdminIdentityResponse', true],
    [
      'LifecycleWriteResult',
      interfaceMembers(adminApiSource, 'LifecycleWriteResult'),
      'LifecycleRecordResponse',
      true,
    ],
    ['RecordListResult', interfaceMembers(adminApiSource, 'RecordListResult'), 'RecordListResponse', true],
    ['QueryResponse', interfaceMembers(apiSource, 'QueryResponse'), 'UnifiedQueryResponse', true],
  ];

  for (const [label, feNames, component, exactBothWays] of PAIRS) {
    it(`${label} fields match schema ${component}`, () => {
      const be = schemaProps(component);
      const missing = feNames.filter((f) => !be.includes(f));
      expect(missing, `FE ${label} reads fields the schema ${component} does not serve`).toEqual([]);
      if (exactBothWays) {
        expect(sorted(feNames), `FE ${label} vs closed schema ${component}`).toEqual(sorted(be));
      }
    });
  }

  it('the FE reads status_code, and the /v2/query schema serves it', () => {
    // Named explicitly because ChatLayout's banner keys off this field; a
    // backend rename here silently kills the warning UI.
    const fe = interfaceMembers(apiSource, 'QueryResponse');
    expect(fe).toContain('status_code');
    expect(schemaProps('UnifiedQueryResponse')).toContain('status_code');
  });
});
