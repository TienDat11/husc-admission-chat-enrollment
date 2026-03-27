# Source Credibility Rankings by Topic

## Software Development / Programming

### Tier 1 — Highest Credibility (Prefer these)
| Domain | Authoritative Sources |
|--------|----------------------|
| Library/Package docs | Official docs site (e.g., react.dev, docs.python.org, nextjs.org) |
| GitHub source code | Official org repos (vercel/next.js, facebook/react, etc.) |
| Language specs | MDN Web Docs, TC39, Rust reference, Go spec |
| Cloud APIs | AWS/GCP/Azure official docs |
| Academic research | arXiv, ACM Digital Library, IEEE Xplore |

### Tier 2 — High Credibility (Good secondary source)
| Domain | Sources |
|--------|---------|
| Framework guides | Official blog posts (e.g., vercel.com/blog, react.dev/blog) |
| OSS examples | GitHub repos with >1000 stars, maintained in last 6 months |
| Community docs | Official wikis (wiki.archlinux.org, etc.) |
| Standards | RFC documents, W3C specs |

### Tier 3 — Medium Credibility (Use with verification)
| Domain | Sources |
|--------|---------|
| Tutorials | Established platforms: egghead.io, Kent C. Dodds blog, Josh W. Comeau |
| Q&A | StackOverflow answers with >50 votes, accepted answers |
| Dev blogs | Company engineering blogs (Stripe, Shopify, Cloudflare engineering) |
| YouTube | Official channel docs/demos only |

### Tier 4 — Low Credibility (Avoid for technical facts)
| Domain | Why to Avoid |
|--------|-------------|
| Random Medium/dev.to posts | Often outdated, no peer review |
| Reddit r/programming | Opinions, not facts |
| Tutorial sites (tutorialspoint, w3schools for advanced) | Often simplified or outdated |
| AI-generated content | Self-referential, potentially hallucinated |

---

## Search Result Quality Signals

### Positive signals (prefer these results):
- URL contains official domain (reactjs.org, docs.rs, pkg.go.dev)
- GitHub URL with starred/maintained repo
- Date within 6 months for fast-moving tech
- Contains code examples that compile/run
- Has version number mentioned explicitly
- References or links to official documentation

### Negative signals (deprioritize or skip):
- "Updated in 2023" for React 19+ questions
- "Tutorial for beginners" when asking about production patterns
- Requires login/account to see content
- URL is a redirect or URL shortener
- Title includes "Top 10 ways" or "Simple guide" for complex technical topics
- No author attribution or date

---

## Topic-Specific Source Priority

### JavaScript/TypeScript Ecosystem
1. MDN Web Docs (JS APIs)
2. Official framework docs (react.dev, nextjs.org, svelte.dev)
3. TypeScript handbook (typescriptlang.org)
4. Node.js docs (nodejs.org)
5. Exa search → GitHub: vercel/\*, facebook/\*, sveltejs/\*

### Python Ecosystem  
1. docs.python.org (language)
2. Official library docs (fastapi.tiangolo.com, sqlalchemy.org)
3. PyPI project pages
4. Exa search → GitHub: tiangolo/\*, pallets/\*, encode/\*

### Mobile (Flutter/React Native)
1. flutter.dev/docs (Flutter)
2. reactnative.dev (React Native)
3. pub.dev (Flutter packages)
4. Exa search → GitHub: flutter/\*, facebook/react-native

### DevOps / Cloud
1. AWS/GCP/Azure official docs
2. CNCF project docs (kubernetes.io, helm.sh)
3. Docker docs (docs.docker.com)
4. Exa search → GitHub: kubernetes/\*, helm/\*

### AI/ML Research
1. arXiv.org (papers)
2. Hugging Face docs (huggingface.co/docs)
3. Papers With Code (paperswithcode.com)
4. Anthropic/OpenAI/Google research blogs
5. Perplexity sonar-reasoning-pro for synthesis

---

## Freshness Requirements by Query Type

| Query Type | Max Age | Why |
|------------|---------|-----|
| "Latest" / "current" / "2026" | 3 months | Time-sensitive |
| Security vulnerabilities | 1 week | Critical |
| API docs for active library | 1 year | APIs change |
| Architectural patterns | 3 years | Slow-moving |
| Algorithm explanations | No limit | Timeless |
| Pricing / SLA / limits | 1 month | Changes frequently |
| Language specs | 2 years | Versioned, stable |

**Implementation**: Use Perplexity `search_recency_filter` for time-sensitive queries.
Available values: `"hour"`, `"day"`, `"week"`, `"month"`, `"year"`
