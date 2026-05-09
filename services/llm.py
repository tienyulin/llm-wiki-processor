import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)

MINIMAX_API_URL = "https://api.minimax.io/v1/text/chatcompletion_v2"
MINIMAX_MODEL = "MiniMax-M2.7"

WIKI_GENERATION_SYSTEM_PROMPT = """You are generating a Karpathy-style wiki structure from markdown documentation.

## Output Structure

Generate markdown files organized in this directory structure:

```
overview.md              # Project overview (required)
llms.txt                 # Index and usage guide (required)
api/
  ├── users.md           # Each module in separate file
  ├── orders.md
  └── inventory.md
architecture/
  ├── system.md
  └── data-flow.md
workflows/
  ├── generation.md
  └── incremental-update.md
concepts/
  ├── karpathy-wiki.md
  └── snapshot.md
guides/
  └── extending-api.md
```

## Every File: YAML Frontmatter

Every markdown file MUST start with YAML frontmatter:

```yaml
---
title: "Users API Module"
type: "api_module"           # Must be one of: api_module, concept, workflow, guide, architecture, overview
module: "users"              # Required for api_module only
description: "User management and authentication APIs"

# For API modules only - list all endpoints
endpoints:
  - method: "GET"
    path: "/users"
    summary: "List all users"
    tags: ["list", "users", "public"]
  - method: "POST"
    path: "/users"
    summary: "Create a new user"
    tags: ["create", "users"]

# For all files - cross-references
related:
  - "api/orders.md"
  - "workflows/authentication.md"
  - "concepts/user-model.md"

tags: ["api", "users"]
last_updated: "2025-02-15T10:30:00Z"
---
```

Type must be one of: api_module, concept, workflow, guide, architecture, overview.
module and endpoints are required for api_module type only.
Every file must have: title, type, description, related, tags, last_updated.

## Content Format

After frontmatter, use standard Markdown. Key rules:

### 1. Wikilinks for Cross-References

Use `[[filename without .md]]` format:

```markdown
# Users Module

This module is used by [[Orders Module]] and [[Inventory Module]].

For authentication flow, see [[Authentication Workflow]].

Related concept: [[User Data Model]]
```

**Important:** Link to actual files that exist.

### 2. Self-Contained Sections

Each file should have clear sections with examples and details.

### 3. Consistent Tags

Use standard tags (lowercase, hyphenated):
- Operations: create, read, list, update, delete, search
- Domains: users, orders, inventory
- Properties: public, internal, deprecated

## Generation Rules

### Rule 1: Separate Files by Concern
- ❌ Wrong: All APIs in one file
- ✅ Right: api/users.md, api/orders.md, api/inventory.md

### Rule 2: Consistent Frontmatter
Every file must have these fields:
- `title` (string)
- `type` (one of: api_module, concept, workflow, guide, architecture, overview)
- `description` (one sentence)
- `related` (array of file references)
- `last_updated` (ISO 8601 format)

For `type: api_module` also require:
- `module` (module name)
- `endpoints` (array with method, path, summary, tags)

### Rule 3: Valid Wikilinks
- Use `[[Name]]` format
- Name should be descriptive but map to actual files
- Examples: `[[Users Module]]` → api/users.md, `[[System Architecture]]` → architecture/system.md

### Rule 4: Hierarchical Organization
Group by semantic type:
- **api/** → All API endpoints
- **architecture/** → System design
- **workflows/** → How things work internally
- **concepts/** → Key ideas and patterns
- **guides/** → How to extend and use

### Rule 5: Output Format

Return files using EXACTLY this format:

```
=== FILE: overview.md ===
---
title: "Project Name"
type: "overview"
description: "Project description"
related: []
tags: ["overview"]
last_updated: "2025-02-15T10:30:00Z"
---

# Project Name

Content here...

=== END FILE ===

=== FILE: api/users.md ===
---
title: "Users API Module"
type: "api_module"
module: "users"
description: "User management APIs"
endpoints:
  - method: "GET"
    path: "/users"
    summary: "List all users"
    tags: ["list", "public"]
related:
  - "api/orders.md"
tags: ["api", "users"]
last_updated: "2025-02-15T10:30:00Z"
---

Content here...

=== END FILE ===
```

## Validation Checklist

Before returning files, verify:
- [ ] Every file has complete frontmatter
- [ ] No missing required fields (title, type, description, last_updated, related)
- [ ] All wikilinks use `[[ ]]` format
- [ ] No dead links (referenced files should exist)
- [ ] api_module files have `module` and `endpoints` array
- [ ] Consistent file organization (api/, architecture/, etc.)
- [ ] Markdown is well-formatted
- [ ] Tags are lowercase and hyphenated
- [ ] last_updated is current date
"""


class MinimaxClient:
    """Client for the Minimax LLM API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.mock_mode = os.getenv("MOCK_LLM", "").lower() in ("true", "1", "yes")
        if self.mock_mode:
            logger.warning("Running in MOCK mode - LLM calls will return mock responses")

    async def _call(self, prompt: str, temperature: float) -> str:
        """Make a single call to the Minimax API and return the assistant message content."""
        if self.mock_mode:
            return self._mock_response()

        async with httpx.AsyncClient(verify=False) as client:
            response = await client.post(
                MINIMAX_API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": MINIMAX_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

    def _mock_response(self) -> str:
        """Return a mock multifile wiki response for testing."""
        return """\
=== FILE: overview.md ===
---
title: "Project Overview"
type: "overview"
description: "Project overview"
related: ["api/users.md", "api/orders.md"]
tags: ["overview"]
last_updated: "2026-01-01T00:00:00Z"
---

# Project Overview

See [[Users API Module]] and [[Orders API Module]].

=== END FILE ===

=== FILE: llms.txt ===
---
title: "Wiki Index"
type: "overview"
description: "Index and usage guide for LLMs"
related: []
tags: ["index"]
last_updated: "2026-01-01T00:00:00Z"
---

# Wiki Index

Browse: api/, architecture/, workflows/, concepts/, guides/

=== END FILE ===

=== FILE: api/users.md ===
---
title: "Users API Module"
type: "api_module"
module: "users"
description: "User management APIs"
endpoints:
  - method: "GET"
    path: "/users"
    summary: "List all users"
    tags: ["list", "users"]
  - method: "POST"
    path: "/users"
    summary: "Create user"
    tags: ["create", "users"]
related: ["api/orders.md"]
tags: ["api", "users"]
last_updated: "2026-01-01T00:00:00Z"
---

# Users API

Manages user accounts. See [[Orders API Module]].

=== END FILE ===

=== FILE: api/orders.md ===
---
title: "Orders API Module"
type: "api_module"
module: "orders"
description: "Order management APIs"
endpoints:
  - method: "GET"
    path: "/orders"
    summary: "List all orders"
    tags: ["list", "orders"]
  - method: "POST"
    path: "/orders"
    summary: "Create order"
    tags: ["create", "orders"]
related: ["api/users.md"]
tags: ["api", "orders"]
last_updated: "2026-01-01T00:00:00Z"
---

# Orders API

Manages orders. See [[Users API Module]].

=== END FILE ===
"""

    def _format_markdowns(self, markdowns: dict) -> str:
        """Format input markdowns into a combined prompt string."""
        return "\n\n".join(
            f"## File: {fname}\n{content}" for fname, content in markdowns.items()
        )

    def _parse_multifile_output(self, response: str) -> dict[str, str]:
        """Parse === FILE: xxx === ... === END FILE === blocks into {path: content}."""
        pattern = r"=== FILE: (.+?) ===\n(.*?)=== END FILE ==="
        matches = re.findall(pattern, response, re.DOTALL)
        if not matches:
            raise ValueError("LLM response contains no valid === FILE === blocks")
        return {path.strip(): content.strip() for path, content in matches}

    def _validate_wiki_structure(self, files: dict[str, str]) -> None:
        """Validate required files exist and all files have YAML frontmatter."""
        required = {"overview.md", "llms.txt"}
        missing = required - set(files.keys())
        if missing:
            raise ValueError(f"Missing required wiki files: {missing}")
        for path, content in files.items():
            if not content.startswith("---"):
                raise ValueError(f"File {path} is missing YAML frontmatter")

    async def generate_wiki(self, markdowns: dict) -> dict[str, str]:
        """Generate wiki file structure from markdown collection."""
        input_markdown = self._format_markdowns(markdowns)
        prompt = (
            f"{WIKI_GENERATION_SYSTEM_PROMPT}\n\n"
            f"## Input Documentation\n\n{input_markdown}\n\n"
            "## Your Task\n\n"
            "1. Analyze the input markdown\n"
            "2. Generate well-organized wiki files\n"
            "3. Each file must have complete YAML frontmatter\n"
            "4. Use wikilinks to connect related files\n"
            "5. Follow all generation rules above\n\n"
            "Generate Now - provide all files using === FILE === format."
        )
        logger.info(f"Calling Minimax for initial wiki generation ({len(input_markdown)} chars)")
        response = await self._call(prompt, temperature=0.3)
        files = self._parse_multifile_output(response)
        self._validate_wiki_structure(files)
        logger.info(f"Successfully generated wiki: {len(files)} files")
        return files

    async def update_wiki(
        self,
        current_files: dict[str, str],
        changed_markdowns: dict,
        changes: dict,
    ) -> dict[str, str]:
        """Incremental update - regenerate affected files, preserve unchanged."""
        changed_content = self._format_markdowns(changed_markdowns)
        current_summary = "\n".join(f"- {path}" for path in sorted(current_files.keys()))
        prompt = (
            f"{WIKI_GENERATION_SYSTEM_PROMPT}\n\n"
            f"## Current Wiki Files\n\n{current_summary}\n\n"
            f"## Changes to Source Documentation\n\n{changes}\n\n"
            f"## New/Modified Source Documentation\n\n{changed_content}\n\n"
            "## Your Task\n\n"
            "1. For added/modified source files: update or create the relevant wiki files\n"
            "2. For deleted source files: remove the relevant wiki sections\n"
            "3. Preserve existing wiki files not affected by these changes\n"
            "4. Return ALL wiki files (unchanged + updated) in === FILE === format\n\n"
            "Generate Now - provide all files using === FILE === format."
        )
        logger.info(f"Calling Minimax for incremental update (changes: {changes})")
        response = await self._call(prompt, temperature=0.2)
        files = self._parse_multifile_output(response)
        self._validate_wiki_structure(files)
        logger.info(f"Successfully updated wiki: {len(files)} files")
        return files
