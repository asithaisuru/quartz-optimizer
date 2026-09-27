"""Expert evidence must classify existing physical leaves without editing results."""
import copy
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import uuid

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from defect_review import atomic_json
from preform_api import create_router


def result_fixture(run_id):
    cuts = [
        {"cut_id":"C1","sequence":1,"parent_piece_id":"P0","result_piece_ids":["P1","P2"],
         "recommendation_status":"selected_verified","manufacturing_verified":True},
        {"cut_id":"C2","sequence":2,"parent_piece_id":"P2","result_piece_ids":["P3","P4"],
         "recommendation_status":"selected_verified","manufacturing_verified":True},
    ]
    pieces = []
    for index, (pid, weight, auto) in enumerate([("P1",20,True),("P3",50,False),("P4",28,False)],1):
        pieces.append({"region_id":f"R{index}","physical_piece_id":pid,"physical_weight_ct":weight,
                       "usable":auto,"usability_status":"usable_preform" if auto else "requires_separation",
                       "morphology":"pointed" if index==2 else "irregular",
                       "suggested_finish_shapes":["pear","kite_diamond_preform"],
                       "mesh_file":"D:/PRIVATE/not-a-public-url.ply"})
    return {
        "run_id":run_id,"recovery_model_version":"v2_usable_preform",
        "stock_partition_basis":"one_original_physical_stock_and_verified_cuts_only",
        "physical_piece_graph":{"root_piece_id":"P0","initial_piece_count":1,
                                "leaf_piece_ids":["P1","P3","P4"],"partitions":[]},
        "physical_piece_count":3,"cuts":cuts,
        "manufacturing_plan":{"status":"complete","diagnostics":{"exact_sequence_verified":True}},
        "recovery_accounting":{"mass_balance_valid":True},
        "regions":pieces,"discarded_regions":[],
        "rough_weight_ct":100,"physical_retained_weight_ct":98,"physical_retention_percent":98,
        "usable_preform_weight_ct":20,"usable_preform_recovery_percent":20,
        "whole_rough_usable":False,"target_applicable":True,"target_recovery_percent":85,
        "candidate_regions":[{"candidate_region_id":"CAND1","physical_weight_ct":900}],
        "diagnostics":{"physical_input_components":[{"component_id":700}]},
    }


@pytest.fixture
def env(tmp_path):
    job=tmp_path/"job"; run_id=uuid.uuid4().hex
    run=job/"preform_recovery"/run_id; run.mkdir(parents=True)
    result=result_fixture(run_id)
    atomic_json(run/"result.json",result)
    atomic_json(run/"status.json",{"run_id":run_id,"status":"completed"})
    for i in range(1,4):
        (run/f"R{i}.ply").write_text("ply\n")
    (job/"analysis_report.json").write_text('{"yield_percent":26.9}')
    app=FastAPI()
    def validate(job_id):
        if job_id != "job":
            raise HTTPException(404,"Job not found.")
        return job
    app.include_router(create_router(validate))
    with TestClient(app) as client:
        yield {"job":job,"run":run,"client":client,"result":result,
               "url":f"/jobs/job/preform-recovery/{run_id}/expert-review"}


def get(e):
    response=e["client"].get(e["url"])
    assert response.status_code==200,response.text
    return response.json()


def decide(e,piece,decision,**extra):
    response=e["client"].patch(e["url"]+"/pieces/"+piece,json={"decision":decision,**extra})
    assert response.status_code==200,response.text
    return response.json()


def test_get_initializes_only_actual_leaves_and_preserves_result(env):
    before=(env["run"]/"result.json").read_bytes()
    r=get(env)
    assert [p["piece_id"] for p in r["pieces"]]==["P1","P3","P4"]
    assert r["pieces"][0]["review_required"] is False
    assert r["pieces"][1]["review_required"] is True
    assert r["pieces"][1]["decision"]=="pending"
    assert r["pieces"][1]["parent_piece_id"]=="P2"
    assert r["pieces"][1]["created_by_cut_id"]=="C2"
    assert r["reviewer"]=={"name":"","code":None,"experience_years":None}
    assert r["summary"]["pending_review_weight_ct"]==78
    assert r["summary"]["review_adjusted_usable_weight_ct"]==20
    assert r["summary"]["target_status"]=="pending_review"
    assert r["summary"]["review_required_count"]==2
    assert not r["summary"]["review_complete"]
    assert (env["run"]/"expert_review.json").is_file()
    assert r["result_sha256"]==hashlib.sha256(before).hexdigest()
    assert (env["run"]/"result.json").read_bytes()==before


def test_all_decisions_reset_formula_and_target_transitions(env):
    r=decide(env,"P3","usable_preform",reason_code="shape_usable",notes="Expert observation")
    assert r["summary"]["expert_confirmed_additional_usable_weight_ct"]==50
    assert r["summary"]["review_adjusted_usable_weight_ct"]==70
    assert r["summary"]["review_adjusted_usable_recovery_percent"]==70
    assert r["summary"]["target_status"]=="pending_review"
    r=decide(env,"P4","usable_preform")
    assert r["summary"]["target_status"]=="met"
    assert r["summary"]["review_complete"]
    assert r["summary"]["reviewed_count"]==2
    r=decide(env,"P3","pending")
    assert r["summary"]["review_adjusted_usable_weight_ct"]==48
    assert r["summary"]["pending_review_weight_ct"]==50
    assert next(p for p in r["pieces"] if p["piece_id"]=="P3")["reviewed_at"] is None
    r=decide(env,"P3","needs_further_separation",reason_code="requires_additional_cut")
    assert r["summary"]["needs_further_separation_weight_ct"]==50
    assert r["summary"]["expert_unusable_weight_ct"]==0
    assert r["summary"]["pending_review_weight_ct"]==0
    assert r["summary"]["target_status"]=="not_met"
    assert r["summary"]["review_complete"]
    assert r["summary"]["physical_retained_weight_ct"]==98
    r=decide(env,"P3","waste_unusable",reason_code="commercially_impractical")
    assert r["summary"]["expert_unusable_weight_ct"]==50
    assert r["summary"]["needs_further_separation_weight_ct"]==0
    assert r["summary"]["review_adjusted_usable_weight_ct"]==48
    assert get(env)==r


def test_met_precedes_pending_when_configured_target_already_reached(env):
    env["result"]["target_recovery_percent"]=60
    atomic_json(env["run"]/"result.json",env["result"])
    r=decide(env,"P3","usable_preform")
    assert r["summary"]["target_status"]=="met"
    assert not r["summary"]["review_complete"]


def test_confirmed_defect_target_stays_not_applicable(env):
    env["result"]["target_applicable"]=False
    atomic_json(env["run"]/"result.json",env["result"])
    assert get(env)["summary"]["target_status"]=="not_applicable"
    assert decide(env,"P3","usable_preform")["summary"]["target_status"]=="not_applicable"


def test_reviewer_metadata_partial_updates_persist(env):
    response=env["client"].patch(env["url"],json={"reviewer":{"name":"Workshop alias","code":"EXP-01","experience_years":15}})
    assert response.status_code==200
    value=response.json()
    assert value["reviewer"]["experience_years"]==15
    response=env["client"].patch(env["url"],json={"reviewer":{"name":""}})
    assert response.json()["reviewer"]=={"name":"","code":"EXP-01","experience_years":15}
    assert get(env)["created_at"]==value["created_at"]
    assert get(env)["reviewer"]==response.json()["reviewer"]


@pytest.mark.parametrize("piece",["unknown","P0","P2","R2","CAND1","700"])
def test_unknown_nonleaf_and_component_ids_rejected(env,piece):
    response=env["client"].patch(env["url"]+"/pieces/"+piece,json={"decision":"usable_preform"})
    assert response.status_code==404


def test_auto_usable_piece_cannot_be_overridden_or_double_counted(env):
    response=env["client"].patch(env["url"]+"/pieces/P1",json={"decision":"waste_unusable"})
    assert response.status_code==409
    assert get(env)["summary"]["auto_validated_usable_weight_ct"]==20


def test_discarded_leaf_is_visible_but_cannot_add_expert_usable_mass(env):
    piece=env["result"]["regions"].pop()
    piece["region_id"]="W3"
    env["result"]["discarded_regions"]=[piece]
    env["result"]["physical_retained_weight_ct"]=70
    env["result"]["physical_retention_percent"]=70
    atomic_json(env["run"]/"result.json",env["result"])
    r=get(env)
    assert len(r["pieces"])==3
    assert r["pieces"][-1]["physically_retained"] is False
    assert r["pieces"][-1]["review_required"] is False
    assert r["pieces"][-1]["mesh_file"] is None
    response=env["client"].patch(env["url"]+"/pieces/P4",json={"decision":"usable_preform"})
    assert response.status_code==409


def test_safe_mesh_url_and_missing_mesh_metadata_fallback(env):
    r=get(env)
    assert r["pieces"][1]["mesh_file"]=="/files/job/preform_recovery/"+env["run"].name+"/R2.ply"
    assert "PRIVATE" not in json.dumps(r)
    (env["run"]/"R2.ply").unlink()
    assert get(env)["pieces"][1]["mesh_file"] is None
    assert decide(env,"P3","usable_preform")["summary"]["review_adjusted_usable_weight_ct"]==70


def test_result_hash_change_is_stale_and_never_applies_old_decisions(env):
    decide(env,"P3","usable_preform")
    evidence=(env["run"]/"expert_review.json").read_bytes()
    env["result"]["diagnostics"]["changed"]=True
    atomic_json(env["run"]/"result.json",env["result"])
    r=get(env)
    assert r["stale"] and r["review_status"]=="stale"
    assert r["summary"]["expert_confirmed_additional_usable_weight_ct"]==0
    assert r["summary"]["pending_review_weight_ct"]==78
    for path,payload in [(env["url"],{"reviewer":{"name":"New"}}),
                         (env["url"]+"/pieces/P3",{"decision":"usable_preform"})]:
        assert env["client"].patch(path,json=payload).status_code==409
    assert (env["run"]/"expert_review.json").read_bytes()==evidence


@pytest.mark.parametrize("payload",[
    {"decision":"invented"},{"decision":"usable_preform","reason_code":"invented"},
    {"decision":"usable_preform","notes":7},{"decision":"usable_preform","weight_ct":90},
    {"decision":"usable_preform","notes":"x"*10001},
])
def test_invalid_piece_payload_rejected(env,payload):
    assert env["client"].patch(env["url"]+"/pieces/P3",json=payload).status_code==422


@pytest.mark.parametrize("payload",[
    {"reviewer":{"experience_years":-1}},{"reviewer":{"experience_years":True}},
    {"reviewer":{"name":None}},{"reviewer":{"unknown":1}},{"piece_reviews":{}},
])
def test_invalid_reviewer_payload_rejected(env,payload):
    assert env["client"].patch(env["url"],json=payload).status_code==422


def test_missing_job_run_and_incomplete_run(env):
    assert env["client"].get(env["url"].replace("/job/","/absent/")).status_code==404
    assert env["client"].get(env["url"].replace(env["run"].name,"invalid-run")).status_code==404
    assert env["client"].get(env["url"].replace(env["run"].name,uuid.uuid4().hex)).status_code==404
    atomic_json(env["run"]/"status.json",{"run_id":env["run"].name,"status":"running"})
    assert env["client"].get(env["url"]).status_code==409


def test_incompatible_v1_result_is_rejected_without_rewrite(env):
    env["result"]["recovery_model_version"]="v1"
    atomic_json(env["run"]/"result.json",env["result"])
    before=(env["run"]/"result.json").read_bytes()
    assert env["client"].get(env["url"]).status_code==422
    assert (env["run"]/"result.json").read_bytes()==before


@pytest.mark.parametrize("damage",["nonleaf","unverified","duplicate","weight"])
def test_invalid_physical_plan_is_rejected(env,damage):
    r=env["result"]
    if damage=="nonleaf":r["physical_piece_graph"]["leaf_piece_ids"]=["P1","P2","P4"]
    if damage=="unverified":r["cuts"][0]["manufacturing_verified"]=False
    if damage=="duplicate":r["regions"][1]["physical_piece_id"]="P1"
    if damage=="weight":r["regions"][1]["physical_weight_ct"]=500
    atomic_json(env["run"]/"result.json",r)
    response=env["client"].get(env["url"])
    assert response.status_code==409
    assert str(env["run"]) not in response.text


def test_all_review_actions_leave_optimizer_and_legacy_bytes_unchanged(env):
    paths=[env["run"]/"result.json",env["run"]/"status.json",env["job"]/"analysis_report.json"]
    before={p:p.read_bytes() for p in paths}
    get(env)
    for decision in ["usable_preform","pending","needs_further_separation","waste_unusable"]:
        decide(env,"P3",decision)
    assert all(p.read_bytes()==data for p,data in before.items())


def test_concurrent_reviews_preserve_both_piece_updates(env):
    get(env)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses=list(pool.map(lambda pid: env["client"].patch(env["url"]+"/pieces/"+pid,
                             json={"decision":"usable_preform"}),["P3","P4"]))
    assert all(r.status_code==200 for r in responses)
    assert get(env)["summary"]["review_adjusted_usable_weight_ct"]==98


def test_original_uncut_physical_leaf_may_be_explicitly_reviewed(env):
    r=env["result"]
    r["cuts"]=[]
    r["manufacturing_plan"]=None
    r["physical_piece_count"]=1
    r["physical_piece_graph"]["leaf_piece_ids"]=["P0"]
    r["regions"]=[{**r["regions"][1],"physical_piece_id":"P0","physical_weight_ct":100}]
    r["physical_retained_weight_ct"]=r["physical_retention_percent"]=100
    r["usable_preform_weight_ct"]=r["usable_preform_recovery_percent"]=0
    atomic_json(env["run"]/"result.json",r)
    review=get(env)
    assert len(review["pieces"])==1
    assert review["pieces"][0]["parent_piece_id"] is None
    assert review["pieces"][0]["created_by_cut_id"] is None
    assert review["summary"]["review_adjusted_usable_weight_ct"]==0
    assert decide(env,"P0","usable_preform")["summary"]["review_adjusted_usable_weight_ct"]==100


def test_invalid_server_metadata_never_leaks_a_filesystem_path(env):
    env["result"]["regions"][0]["usability_status"]="C:/PRIVATE/server/file"
    atomic_json(env["run"]/"result.json",env["result"])
    response=env["client"].get(env["url"])
    assert response.status_code==409
    assert "PRIVATE" not in response.text


def test_persisted_review_is_readable_through_a_new_app_instance(env):
    decide(env,"P3","usable_preform",reason_code="geometry_usable",notes="Persisted evidence")
    app=FastAPI()
    app.include_router(create_router(lambda job_id:env["job"]))
    with TestClient(app) as client:
        response=client.get(env["url"])
    assert response.status_code==200
    assert response.json()["summary"]["expert_confirmed_additional_usable_weight_ct"]==50
