# HUSC uni-guide-ai — React/Vite frontend

# Stage 1: build the static bundle with node:24-alpine.
#
# prebuild (npm run build) runs `types:gen` first, which HARD-ERRORS if
# ../../api/openapi.json (repo root: api/openapi.json) is missing. The schema
# is committed to the repo (regenerate with rag2025/scripts/export_openapi.py).
#
# Build with Docker BuildKit + additional_contexts (see docker-compose.yml):
#   additional_contexts:
#     repo: <repo root>
# so COPY --from=repo can pull in the schema without including rag2025/data/.

FROM node:24-alpine AS build

WORKDIR /app/uni-guide-ai

# OpenAPI schema → /app/api/openapi.json
# types:gen resolves here/../../api/openapi.json relative to scripts/ = /app/api/openapi.json ✓
COPY --from=repo api/openapi.json /app/api/openapi.json

# Deps first for layer caching
COPY package.json package-lock.json ./
# npm ci refuses this lock: it was generated on a platform whose optional-dep
# entries omit @rollup/rollup-linux-x64-gnu and @swc/core-linux-x64-gnu, and
# npm 11 treats that as "lock out of sync". npm install reads the same lock for
# every pinned version and adds the missing platform binaries for this image.
RUN npm install --no-audit --no-fund

# Rest of the FE source
COPY . .

# PROD build with same-origin API base (Caddy reverse-proxies API → BE).
# VITE_API_BASE_URL="" → getApiBaseUrl() returns '' in PROD, no hardcoded URL.
ENV VITE_API_BASE_URL=""
RUN npm run build

# ── Stage 2: Caddy serves the static bundle + proxies API calls to BE ───────
FROM caddy:2.9-alpine

COPY --from=build /app/uni-guide-ai/dist /srv/husc

# Caddyfile is baked in from the repo-root additional context.
# DOMAIN is overridden at runtime via compose environment:
#   DOMAIN=<prod-domain>  → Caddy auto-issues TLS cert via ACME
#   (unset/localhost)    → Caddy uses internal CA on :443
COPY --from=repo rag2025/Caddyfile /etc/caddy/Caddyfile

EXPOSE 80 443
