"""Opt-in HTTP expert-review smoke; decisions are simulated on scratch copies only."""
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import uuid

import requests

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"backend/tests"))
from smoke_preform_real import SOURCES, digest, assert_public


def main():
    source_summary=REPO/"tmp/preform_real_smoke_20260928_013242_06c4e9/smoke_summary.json"
    completed=json.loads(source_summary.read_text())["specimens"]
    scratch=REPO/"tmp"/("preform_expert_smoke_"+time.strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:6])
    scratch.mkdir(parents=True)
    inventories={str(source):{str(p.relative_to(source)):(p.stat().st_size,p.stat().st_mtime_ns)
                             for p in source.rglob("*") if p.is_file()}
                 for source in SOURCES.values()}
    originals={}
    cases={}
    for name,entry in completed.items():
        source_run=source_summary.parent/"jobs"/entry["scratch_job_id"]/"preform_recovery"/entry["run_id"]
        job=scratch/"jobs"/str(uuid.uuid4())
        run=job/"preform_recovery"/entry["run_id"]
        run.mkdir(parents=True)
        for file in [source_run/"result.json",source_run/"status.json",*source_run.glob("R*.ply")]:
            originals[str(file)]=digest(file)
            shutil.copy2(file,run/file.name)
        for relative in ["analysis_report.json","dense/final_textured_model.ply",
                         "extended_search/results.json","extended_search/result_v2/analysis_report.json"]:
            original=SOURCES[name]/relative
            originals[str(original)]=digest(original)
        shutil.copy2(SOURCES[name]/"analysis_report.json",job/"analysis_report.json")
        result=json.loads((run/"result.json").read_text())
        cases[name]=(job,run,result)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1",0));port=probe.getsockname()[1]
    base=f"http://127.0.0.1:{port}"
    environment=dict(os.environ,STORAGE_PATH=str(scratch),API_BASE_URL=base)
    summaries={}
    with (scratch/"server.log").open("w",encoding="utf-8") as logs:
        process=subprocess.Popen([sys.executable,"-B","-m","uvicorn","main:app","--app-dir","backend",
                                  "--host","127.0.0.1","--port",str(port),"--log-level","warning"],
                                 cwd=REPO,env=environment,stdout=logs,stderr=logs,
                                 creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        try:
            deadline=time.monotonic()+40
            while time.monotonic()<deadline:
                try:
                    if requests.get(base+"/optimization-modes",timeout=1).status_code==200:
                        break
                except requests.RequestException:
                    pass
                if process.poll() is not None:raise RuntimeError("Scratch HTTP server exited.")
                time.sleep(.1)
            else:raise TimeoutError("Scratch server not ready.")
            with requests.Session() as client:
                for name,(job,run,result) in cases.items():
                    url=base+f"/jobs/{job.name}/preform-recovery/{run.name}/expert-review"
                    before=digest(run/"result.json")
                    def request(method,path=url,payload=None):
                        response=client.request(method,path,json=payload,timeout=30)
                        assert response.status_code==200,response.text
                        value=response.json();assert_public(value);return value
                    initial=request("GET")
                    assert len(initial["pieces"])==len(result["physical_piece_graph"]["leaf_piece_ids"])
                    assert {p["piece_id"] for p in initial["pieces"]}==set(result["physical_piece_graph"]["leaf_piece_ids"])
                    assert initial["summary"]["auto_validated_usable_weight_ct"]==result["usable_preform_weight_ct"]
                    pending=[p for p in initial["pieces"] if p["review_required"]]
                    assert pending and all(p["decision"]=="pending" for p in pending)
                    assert math.isclose(initial["summary"]["pending_review_weight_ct"],result["requires_separation_weight_ct"],abs_tol=1e-6)
                    assert initial["summary"]["target_status"]=="pending_review"
                    for piece in initial["pieces"]:
                        if piece["mesh_file"]:
                            response=client.get(base+piece["mesh_file"],timeout=30)
                            assert response.status_code==200 and response.content
                    metadata={"name":"SMOKE TEST ONLY - simulated reviewer","code":"TEST-NOT-EXPERT","experience_years":None}
                    assert request("PATCH",payload={"reviewer":metadata})["reviewer"]==metadata
                    piece=max(pending,key=lambda p:p["weight_ct"])
                    piece_url=url+"/pieces/"+piece["piece_id"]
                    usable=request("PATCH",piece_url,{"decision":"usable_preform","reason_code":"geometry_usable",
                                                     "notes":"SIMULATED TEST ONLY; not an actual expert assessment."})
                    addition=piece["weight_ct"]
                    expected=result["usable_preform_weight_ct"]+addition
                    assert math.isclose(usable["summary"]["review_adjusted_usable_weight_ct"],expected,abs_tol=1e-8)
                    assert math.isclose(usable["summary"]["review_adjusted_usable_recovery_percent"],
                                        expected/result["rough_weight_ct"]*100,abs_tol=1e-8)
                    reset=request("PATCH",piece_url,{"decision":"pending","reason_code":None,"notes":""})
                    assert reset["summary"]==initial["summary"]
                    further=request("PATCH",piece_url,{"decision":"needs_further_separation","reason_code":"requires_additional_cut"})
                    assert further["summary"]["needs_further_separation_weight_ct"]==addition
                    assert further["summary"]["expert_unusable_weight_ct"]==0
                    assert further["summary"]["review_adjusted_usable_weight_ct"]==result["usable_preform_weight_ct"]
                    waste=request("PATCH",piece_url,{"decision":"waste_unusable","reason_code":"other"})
                    assert waste["summary"]["expert_unusable_weight_ct"]==addition
                    assert waste["summary"]["needs_further_separation_weight_ct"]==0
                    assert waste["summary"]["physical_retained_weight_ct"]==result["physical_retained_weight_ct"]
                    final=request("PATCH",piece_url,{"decision":"pending","reason_code":None,"notes":""})
                    assert final["summary"]==initial["summary"]
                    assert request("GET")["summary"]==initial["summary"]
                    assert digest(run/"result.json")==before
                    stored=json.loads((run/"expert_review.json").read_text())
                    assert stored["result_sha256"]==before
                    summaries[name]={"scratch_job_id":job.name,"run_id":run.name,
                                     "physical_leaf_count":len(initial["pieces"]),
                                     "review_required_count":initial["summary"]["review_required_count"],
                                     "initial_summary":initial["summary"],"simulated_piece_id":piece["piece_id"],
                                     "simulated_additional_weight_ct":addition,
                                     "simulated_review_adjusted_percent":usable["summary"]["review_adjusted_usable_recovery_percent"],
                                     "simulated_increase_percentage_points":addition/result["rough_weight_ct"]*100,
                                     "all_four_decisions_verified":True,"final_reset_to_pending":True,
                                     "result_sha256":before,"optimizer_result_unchanged":True,
                                     "all_mesh_downloads_passed":True,"no_server_path_leak":True,
                                     "expert_evidence_claim":False}
                    print(name+": "+json.dumps(summaries[name]),flush=True)
            assert all(digest(Path(path))==sha for path,sha in originals.items())
            for source in SOURCES.values():
                after={str(p.relative_to(source)):(p.stat().st_size,p.stat().st_mtime_ns)
                       for p in source.rglob("*") if p.is_file()}
                assert after==inventories[str(source)]
            (scratch/"smoke_summary.json").write_text(json.dumps({
                "simulation_only":True,"original_jobs_unchanged":True,"source_runs_unchanged":True,
                "source_hashes":originals,"original_job_inventory_count":sum(map(len,inventories.values())),
                "specimens":summaries},indent=2),encoding="utf-8")
            print("SMOKE_REPORT="+str(scratch/"smoke_summary.json"),flush=True)
        finally:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill();process.wait(timeout=5)


if __name__=="__main__":
    main()
