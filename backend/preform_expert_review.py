"""Additive expert evidence for immutable, completed V2 physical cut-tree leaves."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import uuid
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Body, HTTPException
from defect_review import atomic_json, finite, now, read_json

DECISIONS = {"pending", "usable_preform", "needs_further_separation", "waste_unusable"}
REASONS = {"shape_usable", "geometry_usable", "requires_additional_cut", "too_small",
           "defect_concern", "handling_concern", "commercially_impractical", "other"}
AUTO_STATUSES = {"usable_preform", "irregular_preform", "requires_separation",
                 "requires_further_separation", "manufacturing_invalid",
                 "defect_constrained", "too_small", "numerical_debris"}
MORPHOLOGIES = {"pointed", "elongated", "blocky", "rounded", "irregular"}
SHAPES = {"pear", "marquise", "kite_diamond_preform", "emerald", "cushion",
          "princess", "round", "oval"}
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}")
DEFAULT_REVIEWER = {"name": "", "code": None, "experience_years": None}


def _identifier(value):
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ValueError("Invalid physical-piece identifier.")
    return value


def _text(value, label, limit):
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(label + " must be text within its length limit.")
    return value


def _reviewer(payload, previous):
    if not isinstance(payload, dict) or set(payload) - set(DEFAULT_REVIEWER):
        raise ValueError("Unsupported reviewer metadata.")
    value = {**previous, **payload}
    _text(value["name"], "Reviewer name", 200)
    if value["code"] is not None:
        _text(value["code"], "Reviewer code", 100)
    if value["experience_years"] is not None:
        finite(value["experience_years"], "experience_years", 0)
    return value


def _decision(payload, previous=None):
    if not isinstance(payload, dict) or set(payload) - {"decision", "reason_code", "notes"}:
        raise ValueError("Unsupported piece review fields.")
    decision = payload.get("decision")
    if not isinstance(decision, str) or decision not in DECISIONS:
        raise ValueError("Unsupported expert decision.")
    value = {**(previous or {"reason_code": None, "notes": ""}), **payload}
    reason = value["reason_code"]
    if reason is not None and (not isinstance(reason, str) or reason not in REASONS):
        raise ValueError("Unsupported expert reason code.")
    _text(value["notes"], "Review notes", 10000)
    value["reviewed_at"] = None if decision == "pending" else now()
    return value


def _mesh_url(job, run, region):
    # Reuse only this run's existing region export; never relay a stored path/URL.
    region_id = region.get("region_id")
    if not isinstance(region_id, str) or not re.fullmatch(r"[RW][1-9][0-9]*", region_id):
        return None
    asset = run / (region_id + ".ply")
    if not asset.resolve().is_relative_to(run.resolve()) or not asset.is_file():
        return None
    return "/files/" + quote(job.name, safe="") + "/preform_recovery/" + run.name + "/" + asset.name


def physical_leaves(result, job, run):
    """Reconstruct physical authority from cuts; labels/candidates never authorize IDs."""
    if (result.get("recovery_model_version") != "v2_usable_preform"
            or result.get("stock_partition_basis") != "one_original_physical_stock_and_verified_cuts_only"):
        raise HTTPException(422, "Expert review requires a physical-piece-compatible V2 result.")
    graph = result["physical_piece_graph"]
    root = _identifier(graph["root_piece_id"])
    if graph["initial_piece_count"] != 1:
        raise ValueError("Invalid initial physical stock.")
    active, seen, provenance = {root}, {root}, {root: (None, None)}
    cuts = result["cuts"]
    plan = result.get("manufacturing_plan") or {}
    if cuts and (plan.get("status") != "complete"
                 or plan.get("diagnostics", {}).get("exact_sequence_verified") is not True):
        raise ValueError("Selected physical cuts are not verified.")
    for index, cut in enumerate(cuts, 1):
        parent = _identifier(cut["parent_piece_id"])
        children = cut["result_piece_ids"]
        if (parent not in active or not isinstance(children, list) or len(children) != 2
                or cut.get("sequence") != index
                or cut.get("recommendation_status") != "selected_verified"
                or cut.get("manufacturing_verified") is not True):
            raise ValueError("Invalid selected physical cut.")
        children = [_identifier(child) for child in children]
        if len(set(children)) != 2 or any(child in seen for child in children):
            raise ValueError("Invalid physical child lineage.")
        active.remove(parent)
        active.update(children)
        seen.update(children)
        for child in children:
            provenance[child] = (parent, _identifier(cut["cut_id"]))
    leaves = graph["leaf_piece_ids"]
    if (len(leaves) != len(active) or set(leaves) != active
            or result["physical_piece_count"] != len(active)
            or len(active) != len(cuts) + 1):
        raise ValueError("Physical leaf graph does not match selected cuts.")
    if not result["recovery_accounting"]["mass_balance_valid"]:
        raise ValueError("The result's physical ledger is invalid.")
    regions = {}
    retained_ids = set()
    for retained, items in ((True, result["regions"]), (False, result.get("discarded_regions", []))):
        for region in items:
            identifier = _identifier(region["physical_piece_id"])
            if identifier not in active or identifier in regions:
                raise ValueError("Region is not a unique physical leaf.")
            weight = finite(region["physical_weight_ct"], "physical weight", 0)
            if weight <= 0 or not isinstance(region["usable"], bool):
                raise ValueError("Invalid physical leaf usability.")
            auto = region["usable"]
            if region["usability_status"] not in AUTO_STATUSES:
                raise ValueError("Invalid automatic usability status.")
            if auto and not retained:
                raise ValueError("Discarded material cannot be auto usable.")
            if retained:
                retained_ids.add(identifier)
            parent, created = provenance[identifier]
            regions[identifier] = {
                "piece_id": identifier, "parent_piece_id": parent, "created_by_cut_id": created,
                "weight_ct": weight, "physically_retained": retained,
                "auto_usable": auto,
                "auto_usability_status": region["usability_status"],
                "morphology": region.get("morphology") if region.get("morphology") in MORPHOLOGIES else "irregular",
                "suggested_finish_shapes": [shape for shape in region.get("suggested_finish_shapes", [])
                                            if isinstance(shape, str) and shape in SHAPES],
                "mesh_file": _mesh_url(job, run, region),
                "review_required": retained and not auto,
            }
    if set(regions) != active:
        raise ValueError("Result is missing physical leaf metadata.")
    rough = finite(result["rough_weight_ct"], "rough weight", 0)
    physical = finite(result["physical_retained_weight_ct"], "physical retained weight", 0)
    auto = finite(result["usable_preform_weight_ct"], "auto usable weight", 0)
    tolerance = max(1e-8, rough * 1e-8)
    if (rough <= 0 or auto > physical + tolerance or physical > rough + tolerance
            or abs(math.fsum(regions[i]["weight_ct"] for i in retained_ids) - physical) > tolerance
            or abs(math.fsum(p["weight_ct"] for p in regions.values() if p["auto_usable"]) - auto) > tolerance):
        raise ValueError("Physical leaf weights do not reconcile to the immutable result.")
    if not cuts and result.get("whole_rough_usable") is not True and auto > tolerance:
        raise ValueError("Unusable uncut stock cannot have automatic usable credit.")
    if not isinstance(result["target_applicable"], bool):
        raise ValueError("Invalid target applicability.")
    finite(result["target_recovery_percent"], "target percent", 0, 100)
    return [regions[identifier] for identifier in leaves]


def _response(result, pieces, record, current_hash):
    stale = record["result_sha256"] != current_hash
    reviews = {} if stale else record["piece_reviews"]
    output = []
    for piece in pieces:
        decision = reviews.get(piece["piece_id"], {})
        output.append({**piece, "decision": decision.get("decision", "pending"),
                       "reason_code": decision.get("reason_code"), "notes": decision.get("notes", ""),
                       "reviewed_at": decision.get("reviewed_at")})
    required = [p for p in output if p["review_required"]]
    buckets = {decision: math.fsum(p["weight_ct"] for p in required if p["decision"] == decision)
               for decision in DECISIONS}
    rough, auto = result["rough_weight_ct"], result["usable_preform_weight_ct"]
    adjusted = math.fsum((auto, buckets["usable_preform"]))
    percent = 100 * adjusted / rough
    reviewed = sum(p["decision"] != "pending" for p in required)
    target = ("not_applicable" if not result["target_applicable"] else
              "met" if percent >= result["target_recovery_percent"] else
              "pending_review" if buckets["pending"] > 0 else "not_met")
    return {
        "schema_version": 1, "job_id": record["job_id"], "run_id": record["run_id"],
        "result_sha256": record["result_sha256"], "current_result_sha256": current_hash,
        "stale": stale, "review_status": "stale" if stale else "current",
        "stale_reason": "completed_result_hash_changed" if stale else None,
        "created_at": record["created_at"], "updated_at": record["updated_at"],
        "reviewer": copy.deepcopy(record["reviewer"]),
        "target_applicable": result["target_applicable"],
        "target_recovery_percent": result["target_recovery_percent"],
        "pieces": output,
        "summary": {
            "rough_weight_ct": rough,
            "physical_retained_weight_ct": result["physical_retained_weight_ct"],
            "physical_retention_percent": result["physical_retention_percent"],
            "auto_validated_usable_weight_ct": auto,
            "auto_validated_usable_recovery_percent": result["usable_preform_recovery_percent"],
            "expert_confirmed_additional_usable_weight_ct": buckets["usable_preform"],
            "review_adjusted_usable_weight_ct": adjusted,
            "review_adjusted_usable_recovery_percent": percent,
            "pending_review_weight_ct": buckets["pending"],
            "needs_further_separation_weight_ct": buckets["needs_further_separation"],
            "expert_unusable_weight_ct": buckets["waste_unusable"],
            "review_required_count": len(required), "reviewed_count": reviewed,
            "review_complete": reviewed == len(required), "target_status": target,
        },
    }


def create_router(validate_job, job_lock):
    router = APIRouter(tags=["preform-expert-review"])

    def execute(job_id, run_id, payload=None, piece_id=None):
        job = Path(validate_job(job_id))
        try:
            identifier = uuid.UUID(run_id).hex
        except (ValueError, TypeError, AttributeError):
            raise HTTPException(404, "Preform run not found.")
        run = job / "preform_recovery" / identifier
        if not run.resolve().is_relative_to(job.resolve()) or not run.is_dir():
            raise HTTPException(404, "Preform run not found.")
        try:
            with job_lock(job):
                status = read_json(run / "status.json", {})
                if status.get("status") != "completed" or status.get("run_id") != identifier:
                    raise HTTPException(409, "Expert review requires a completed run.")
                result_bytes = (run / "result.json").read_bytes()
                current_hash = hashlib.sha256(result_bytes).hexdigest()
                result = json.loads(result_bytes)
                if result.get("run_id") != identifier:
                    raise ValueError("Run identity mismatch.")
                pieces = physical_leaves(result, job, run)
                path = run / "expert_review.json"
                record = read_json(path)
                fresh = record is None
                if fresh:
                    timestamp = now()
                    record = {"schema_version": 1, "job_id": job_id, "run_id": identifier,
                              "result_sha256": current_hash, "reviewer": dict(DEFAULT_REVIEWER),
                              "piece_reviews": {}, "created_at": timestamp, "updated_at": timestamp}
                if (record.get("schema_version") != 1 or record.get("job_id") != job_id
                        or record.get("run_id") != identifier):
                    raise ValueError("Invalid stored expert review identity.")
                if (not isinstance(record.get("result_sha256"), str)
                        or not re.fullmatch(r"[0-9a-f]{64}", record["result_sha256"])):
                    raise ValueError("Invalid stored result binding.")
                _reviewer(record["reviewer"], DEFAULT_REVIEWER)
                if not isinstance(record["piece_reviews"], dict):
                    raise ValueError("Invalid stored piece reviews.")
                stale = record["result_sha256"] != current_hash
                lookup = {p["piece_id"]: p for p in pieces}
                if not stale:
                    for stored_id, stored in record["piece_reviews"].items():
                        if stored_id not in lookup or not lookup[stored_id]["review_required"]:
                            raise ValueError("Stored decision does not belong to a review-required leaf.")
                        _decision({k: stored[k] for k in ("decision", "reason_code", "notes")})
                if payload is not None:
                    if stale:
                        raise HTTPException(409, "Expert review is stale: the completed result changed. Use a new completed run.")
                    try:
                        if piece_id is None:
                            if set(payload) != {"reviewer"}:
                                raise ValueError("Only reviewer metadata may be patched here.")
                            record["reviewer"] = _reviewer(payload["reviewer"], record["reviewer"])
                        else:
                            if piece_id not in lookup:
                                raise HTTPException(404, "Physical leaf piece not found.")
                            if not lookup[piece_id]["review_required"]:
                                raise HTTPException(409, "Only unresolved retained physical leaves accept expert decisions.")
                            record["piece_reviews"][piece_id] = _decision(payload, record["piece_reviews"].get(piece_id))
                        record["updated_at"] = now()
                    except ValueError as exc:
                        raise HTTPException(422, str(exc))
                response = _response(result, pieces, record, current_hash)
                if fresh or payload is not None:
                    # Recheck binding immediately before the sole evidence write.
                    if hashlib.sha256((run / "result.json").read_bytes()).hexdigest() != current_hash:
                        raise HTTPException(409, "Completed result changed during expert review.")
                    atomic_json(path, record)
                return response
        except HTTPException:
            raise
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            raise HTTPException(409, "Expert review data is unavailable or incompatible with this completed result.")

    @router.get("/jobs/{job_id}/preform-recovery/{run_id}/expert-review")
    def get_review(job_id: str, run_id: str):
        return execute(job_id, run_id)

    @router.patch("/jobs/{job_id}/preform-recovery/{run_id}/expert-review")
    def patch_session(job_id: str, run_id: str, payload: dict = Body(...)):
        return execute(job_id, run_id, payload)

    @router.patch("/jobs/{job_id}/preform-recovery/{run_id}/expert-review/pieces/{piece_id}")
    def patch_piece(job_id: str, run_id: str, piece_id: str, payload: dict = Body(...)):
        return execute(job_id, run_id, payload, piece_id)

    return router
