"""Tests for MinimaxClient multifile parsing and validation."""
import pytest
from services.llm import MinimaxClient


@pytest.fixture
def client():
    return MinimaxClient(api_key="test-key")


VALID_MULTIFILE = """\
=== FILE: overview.md ===
---
title: "Overview"
type: "overview"
description: "Project overview"
related: []
tags: ["overview"]
last_updated: "2026-01-01T00:00:00Z"
---

# Overview

Content here.

=== END FILE ===

=== FILE: llms.txt ===
---
title: "Index"
type: "overview"
description: "Wiki index"
related: []
tags: ["index"]
last_updated: "2026-01-01T00:00:00Z"
---

# Index

Browse api/, architecture/

=== END FILE ===

=== FILE: api/users.md ===
---
title: "Users API"
type: "api_module"
module: "users"
description: "User APIs"
endpoints: []
related: []
tags: ["api"]
last_updated: "2026-01-01T00:00:00Z"
---

# Users API

=== END FILE ===
"""


def test_parse_multifile_output(client):
    files = client._parse_multifile_output(VALID_MULTIFILE)
    assert "overview.md" in files
    assert "llms.txt" in files
    assert "api/users.md" in files
    assert "# Overview" in files["overview.md"]
    assert "# Users API" in files["api/users.md"]


def test_parse_multifile_output_no_blocks(client):
    with pytest.raises(ValueError, match="no valid"):
        client._parse_multifile_output("This has no file blocks.")


def test_validate_wiki_structure_valid(client):
    files = client._parse_multifile_output(VALID_MULTIFILE)
    client._validate_wiki_structure(files)  # Should not raise


def test_validate_wiki_structure_missing_required(client):
    files = {"api/users.md": "---\ntitle: Users\n---\n# Users"}
    with pytest.raises(ValueError, match="Missing required"):
        client._validate_wiki_structure(files)


def test_validate_wiki_structure_missing_frontmatter(client):
    files = {
        "overview.md": "# No frontmatter here",
        "llms.txt": "---\ntitle: Index\n---\n# Index",
    }
    with pytest.raises(ValueError, match="missing YAML frontmatter"):
        client._validate_wiki_structure(files)


def test_mock_response_is_valid(client):
    import os
    os.environ["MOCK_LLM"] = "true"
    mock_client = MinimaxClient(api_key="test-key")
    response = mock_client._mock_response()
    files = mock_client._parse_multifile_output(response)
    mock_client._validate_wiki_structure(files)
    assert len(files) >= 2
