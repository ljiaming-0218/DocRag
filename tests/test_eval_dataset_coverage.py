from eval.tools.audit_dataset_coverage import build_coverage_report


def test_coverage_report_identifies_category_and_follow_up_gaps():
    cases = [
        {
            "case_id": "FACT-01",
            "document_key": "doc-a",
            "scenario": "single_turn",
            "category": "fact",
            "answerable": True,
            "gold_evidence": [{"evidence_id": "E1"}],
        },
        {
            "case_id": "NOANSWER-01",
            "document_key": "doc-a",
            "scenario": "follow_up",
            "category": "unanswerable",
            "answerable": False,
            "gold_evidence": [],
        },
    ]

    report = build_coverage_report(cases)

    assert report["total_cases"] == 2
    assert report["category_counts"]["fact"] == 1
    assert report["category_gaps"]["fact"] == 2
    assert "follow_up_sample_count" in report["gaps"]
    assert "multi_document_contract" in report["gaps"]


def test_coverage_report_finds_missing_evidence_and_duplicate_ids():
    cases = [
        {
            "case_id": "CASE-01",
            "document_key": "doc-a",
            "scenario": "single_turn",
            "category": "fact",
            "answerable": True,
            "gold_evidence": [],
        },
        {
            "case_id": "CASE-01",
            "document_key": "doc-b",
            "scenario": "single_turn",
            "category": "term",
            "answerable": True,
            "gold_evidence": [{"evidence_id": "E2"}],
        },
    ]

    report = build_coverage_report(cases)

    assert report["missing_gold_evidence"] == ["CASE-01"]
    assert report["duplicate_case_ids"] == ["CASE-01"]
