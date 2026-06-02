from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = PROJECT_ROOT / "data/1_intermediate/claim_grounding/human_audit/human_audit.html"


def test_human_audit_html_is_self_contained() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")

    assert "<!doctype html>" in html.lower()
    assert "script src=" not in html.lower()
    assert "rel=\"stylesheet\"" not in html.lower()
    assert "https://" not in html.lower()


def test_human_audit_html_loads_queue_and_supports_file_fallback() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")

    assert "human_audit_queue.jsonl" in html
    assert "human_audit_summary.json" in html
    assert "queueFileInput" in html
    assert "parseJsonl" in html
    assert "loadBundledAuditData" in html


def test_human_audit_html_saves_progress_and_exports_jsonl() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")

    assert "localStorage" in html
    assert "saveCurrentReview" in html
    assert "downloadResultsJsonl" in html
    assert "human_audit_results.jsonl" in html
    assert "audit_diagnostic_label" in html
    assert "auditor_notes" in html


def test_human_audit_html_has_review_controls_and_filters() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")

    for label in (
        "valid_evidence",
        "passage_quality_issue",
        "retrieval_failure",
        "language_retrieval_failure",
        "ambiguous_claim",
        "model_error",
        "skip",
    ):
        assert f'data-label="{label}"' in html
    for old_label in ("support", "contradict", "context", "insufficient"):
        assert f'data-label="{old_label}"' not in html

    for control_id in ("eventFilter", "flagFilter", "statusFilter", "labelFilter", "searchInput"):
        assert control_id in html


def test_human_audit_html_cleans_raw_passage_fragments_before_display() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")

    assert "function cleanPassageText" in html
    assert "function displayPassageText" in html
    assert "displayPassageText(passage)" in html
    assert "\\bStat es\\b" in html
    assert "\\d+\\/\\d+" in html
