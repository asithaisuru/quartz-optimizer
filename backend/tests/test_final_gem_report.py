"""Final Gemstone Planning Report: one defect-aware run is the only source."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import defect_aware_api as api
import optimizer
from defect_review import atomic_json, read_json
from final_gem_report import (
    EXCLUSION_NOTE, SAFETY_STATEMENT, TITLE, dashboard_rows, final_gem_summary, format_ct,
    format_percent,
)
from test_defect_aware_faceted import annotation, run, service, strategy  # noqa: F401  (fixture)

TWO_GEMS = strategy(((-4, 0, 0), (4, 0, 0)))


def report(client, root):
    return client.get(root + "/report.pdf")


def completed_run(client, candidate=TWO_GEMS):
    with patch.object(optimizer, "optimize_cut", return_value=[candidate]):
        return run(client)


def test_dashboard_values_match_the_frontend_formatting():
    # Same inputs and expected strings as the frontend presentation tests (QZ-05 run).
    rows = dict(dashboard_rows(final_gem_summary({
        "rough_weight_ct": 244.02, "confirmed_defect_count": 1,
        "confirmed_defect_excluded_ct": 8.141048526863086, "total_gem_weight_ct": 54.13529012278915,
        "faceted_yield_percent": 22.18477588836536, "gem_count": 4,
        "manufacturing_status": "complete", "runtime_seconds": 127.34049019999657,
    })))
    assert rows == {
        "Rough Weight": "244.02 ct",
        "Confirmed Defects": "1 confirmed",
        "Confirmed Defect Exclusion - Final Gem Optimizer": "8.14 ct",
        "Clean Rough Weight": "235.88 ct",
        "Final Gemstone Weight": "54.14 ct",
        "Final Faceted Yield": "23.0%",
        "Gem Count": "4",
        "Manufacturing Status": "Complete",
        "Runtime": "2 min 7 s",
    }


def test_endpoint_returns_a_valid_report_for_the_current_run(service):
    job, client = service
    root, result = completed_run(client)
    response = report(client, root)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/pdf"
    assert result["run_id"][:8] in response.headers["content-disposition"]
    pdf = response.content
    assert pdf.startswith(b"%PDF-")
    for text in (TITLE, "testjob", result["run_id"], EXCLUSION_NOTE, SAFETY_STATEMENT,
                 "Rough Stone", "Defect Review", "Final Gemstone Result", "Gemstone Breakdown",
                 "Manufacturing Verification", "Cut Sequence", "Limitations",
                 "Bounded search", "Not measured polished recovery", "Workshop validation remains required"):
        assert text.encode() in pdf, text
    # Every dashboard value for this run appears in the PDF.
    for label, value in dashboard_rows(final_gem_summary(result)):
        assert value.encode() in pdf, (label, value)
    # Gem table and cut sequence rows come from this run.
    for gem in result["gems"]:
        assert gem["id"].encode() in pdf and f"{gem['weight_ct']:.2f}".encode() in pdf
    assert len(result["cut_sequence"]) == 1
    step = result["cut_sequence"][0]
    assert step["parent_piece_id"].encode() in pdf
    assert f"{step['required_depth_mm']:g} mm".encode() in pdf
    assert b"Number of verified cuts" in pdf


def test_report_never_reads_stone_preservation_or_legacy_results(service):
    job, client = service
    # Distinctive values planted where other models store their results.
    atomic_json(job / "analysis_report.json", {"raw_carats": 999.99, "estimated_cut_carats": 888.88,
                                              "yield_percent": 77.77})
    other = job / "stone_preservation" / ("0" * 32)
    other.mkdir(parents=True)
    atomic_json(other / "result.json", {"confirmed_defect_excluded_ct": 97.531,
                                       "clean_material_available_ct": 543.21})
    root, result = completed_run(client)
    pdf = report(client, root).content
    for planted in (b"999.99", b"888.88", b"77.77", b"97.53", b"543.21", b"Stone Preservation",
                    b"Preform Recovery"):
        assert planted not in pdf, planted
    assert format_ct(result["confirmed_defect_excluded_ct"]).encode() in pdf


def test_stale_result_is_refused(service):
    job, client = service
    root, _ = completed_run(client)
    assert report(client, root).status_code == 200
    value = read_json(job / "defect_review.json")
    value["annotations"][0]["geometry"]["radii_mm"][0] = 2
    atomic_json(job / "defect_review.json", value)
    response = report(client, root)
    assert response.status_code == 409
    assert response.json()["detail"] == "Recalculate Final Gemstones before downloading the report."


def test_report_is_cached_per_run_and_regenerated_only_when_the_result_changes(service):
    job, client = service
    root, result = completed_run(client)
    run_dir = job / api.DIRECTORY / result["run_id"]
    with patch.object(api, "create_final_gem_pdf", wraps=api.create_final_gem_pdf) as generate:
        first = report(client, root).content
        assert report(client, root).content == first
        assert generate.call_count == 1
        saved = read_json(run_dir / "result.json")
        saved["runtime_seconds"] = 12.5
        atomic_json(run_dir / "result.json", saved)
        updated = report(client, root).content
        assert generate.call_count == 2
    assert b"12 s" in updated or b"13 s" in updated
    assert (run_dir / api.REPORT_FILENAME).read_bytes() == updated


def test_new_run_gets_its_own_report_and_history_is_kept(service):
    job, client = service
    first_root, first = completed_run(client)
    first_pdf = report(client, first_root).content
    second_root, second = completed_run(client, strategy(((-4, 0, 0), (4, 0, 0), (0, 5, 0))))
    second_pdf = report(client, second_root).content
    assert second["run_id"] != first["run_id"]
    assert second["run_id"].encode() in second_pdf and first["run_id"].encode() not in second_pdf
    assert format_ct(second["total_gem_weight_ct"]).encode() in second_pdf
    # The earlier run's report is not overwritten and still describes that run.
    assert report(client, first_root).content == first_pdf
    assert (job / api.DIRECTORY / first["run_id"] / api.REPORT_FILENAME).read_bytes() == first_pdf


def test_incomplete_empty_and_unknown_runs_have_no_report(service):
    job, client = service
    with patch.object(optimizer, "optimize_cut", side_effect=RuntimeError("search failed")):
        queued = client.post("/jobs/testjob/defect-aware-optimization", json={}).json()
    assert report(client, "/jobs/testjob/defect-aware-optimization/" + queued["run_id"]).status_code == 409
    empty_root, empty = completed_run(client, strategy(((0, 0, 0),)))
    assert empty["gem_count"] == 0
    assert report(client, empty_root).status_code == 409
    assert report(client, "/jobs/testjob/defect-aware-optimization/not-a-run").status_code == 404


def test_other_modes_reusing_the_router_get_no_final_report(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import stone_preservation_api
    app = FastAPI()
    app.include_router(stone_preservation_api.create_router(lambda _: tmp_path))
    paths = {route.path for route in app.routes}
    assert not any(path.endswith("/report.pdf") for path in paths)
    assert "/jobs/{job_id}/stone-preservation/{run_id}/result" in paths
    with TestClient(app) as client:
        assert client.get(f"/jobs/x/stone-preservation/{'0' * 32}/report.pdf").status_code == 404


def test_percent_and_ct_formatting_handle_missing_values():
    assert format_ct(None) == "-" and format_percent(float("nan")) == "-"
    summary = final_gem_summary({"rough_weight_ct": 10, "total_gem_weight_ct": 2})
    assert summary["clean_rough_weight_ct"] is None and summary["clean_yield_percent"] is None
