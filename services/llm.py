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
```

## Every File: YAML Frontmatter

Every markdown file MUST start with YAML frontmatter:

```yaml
---
title: "Users API Module"
type: "api_module"
module: "users"
description: "User management and authentication APIs"
endpoints:
  - method: "GET"
    path: "/users"
    summary: "List all users"
    tags: ["list", "users"]
related:
  - "api/orders.md"
tags: ["api", "users"]
last_updated: "2025-02-15T10:30:00Z"
---
```

Type must be one of: api_module, concept, workflow, guide, architecture, overview.
module and endpoints are required for api_module type only.
Every file must have: title, type, description, related, tags, last_updated.

## Wikilinks

Use [[Title]] format for cross-references. Link only to files that exist.

## Generation Rules

1. Separate files by concern: one file per API module
2. Hierarchical organization: api/, architecture/, workflows/, concepts/, guides/
3. Tags: lowercase, hyphenated

## Output Format

Return files using EXACTLY this format:

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
description: "..."
endpoints: []
related: []
tags: ["api", "users"]
last_updated: "2025-02-15T10:30:00Z"
---

Content here...

=== END FILE ===
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
