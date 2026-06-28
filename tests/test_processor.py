"""Tests for WikiProcessor.detect_changes (pure function, no I/O).

pylint: the `processor` fixture is injected into tests under the same name
(redefined-outer-name) — standard pytest fixture convention.
"""

# pylint: disable=redefined-outer-name

from unittest.mock import MagicMock

import pytest

from services.processor import WikiProcessor


@pytest.fixture
def processor():
    """A WikiProcessor with mock storage/LLM for pure detect_changes tests."""
    storage = MagicMock()
    llm = MagicMock()
    return WikiProcessor(storage=storage, llm=llm)


def test_first_run_all_added(processor):
    """Empty old snapshot → all new files appear in 'added'."""
    new = {"a.md": "content a", "b.md": "content b"}
    result = processor.detect_changes({}, new)
    assert sorted(result["added"]) == ["a.md", "b.md"]
    assert result["modified"] == []
    assert result["deleted"] == []


def test_one_file_added(processor):
    """A single new file appears only in 'added'."""
    old = {"a.md": "content a"}
    new = {"a.md": "content a", "b.md": "content b"}
    result = processor.detect_changes(old, new)
    assert result["added"] == ["b.md"]
    assert result["modified"] == []
    assert result["deleted"] == []


def test_file_content_changed(processor):
    """A changed file's content appears only in 'modified'."""
    old = {"a.md": "old content"}
    new = {"a.md": "new content"}
    result = processor.detect_changes(old, new)
    assert result["added"] == []
    assert result["modified"] == ["a.md"]
    assert result["deleted"] == []


def test_file_removed(processor):
    """A dropped file appears only in 'deleted'."""
    old = {"a.md": "content a", "b.md": "content b"}
    new = {"a.md": "content a"}
    result = processor.detect_changes(old, new)
    assert result["added"] == []
    assert result["modified"] == []
    assert result["deleted"] == ["b.md"]


def test_no_changes(processor):
    """Identical old/new snapshots yield empty added/modified/deleted."""
    snap = {"a.md": "same", "b.md": "same"}
    result = processor.detect_changes(snap, dict(snap))
    assert result["added"] == []
    assert result["modified"] == []
    assert result["deleted"] == []
