"""Only selected physical cuts may create preform children."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preform_material import VoxelStock
from preform_features import component_candidates
from preform_recovery import DEFAULTS, morphology
from preform_usability import evaluate_usability, plan_score
from test_preform_usability import cells, physical
from test_preform_material import split_plan


def assess(result):
    return evaluate_usability(result,DEFAULTS,1,morphology)


def test_reconstruction_labels_do_not_split_original_or_discard_tiny_candidates():
    stock=cells((10,5,5))
    stock.source_labels=np.zeros(len(stock.active),int)
    stock.source_labels[-1]=1  # Tiny modeled island is not a physical discard.
    result=physical(stock)
    check=assess(result)
    assert len(result["pieces"])==1
    assert result["natural_components"]==[]
    assert result["discarded"]==0
    assert result["retained"]==100
    assert check["usable_preform_weight_ct"]==0
    assert result["balance"]["mass_balance_error_ct"]==0


def test_verified_cut_creates_exactly_two_children_plus_kerf_despite_many_labels():
    stock=cells((10,5,5))
    stock.source_labels=stock.indices[:,0]//2
    result=physical(stock,split_plan(offset=4.5),kerf=.2)
    assert len(result["pieces"])==2
    assert len(result["partitions"])==1
    assert result["natural_components"]==[]
    assert abs(result["retained"]+result["kerf"]-100)<1e-8
    assert abs(result["kerf"]-2)<1e-8
    for piece in result["pieces"].values():
        assert piece["parent_piece_id"]=="rough_piece_1"
        assert piece["created_by_cut_step"]==1
    check=assess(result)
    assert all("usability" in p for p in result["pieces"].values())
    assert check["usable_preform_weight_ct"]==0  # No arbitrary within-child label credit.


def pointed_stock():
    body=np.indices((6,7,7)).reshape(3,-1).T
    neck=np.array([[x,3,3] for x in range(6,10)])
    tip=[]
    for x in range(20):
        radius=max(1,(19-x)//3)
        for y in range(-radius,radius+1):
            for z in range(-radius,radius+1):
                if y*y+z*z<=radius*radius:
                    tip.append([10+x,3+y,3+z])
    indices=np.vstack([body,neck,tip])
    stock=VoxelStock(indices,np.zeros(3),1)
    stock.source_labels=(indices[:,0]>=10).astype(int)
    return stock


def test_pointed_candidate_is_not_credited_until_verified_cut_isolates_it():
    stock=pointed_stock()
    original=physical(stock)
    original_check=assess(original)
    assert len(original["pieces"])==1
    assert original_check["usable_preform_weight_ct"]==0
    proposals=component_candidates(stock,.5,100/len(stock.active))
    assert proposals and all(p["requires_verified_cut"] for p in proposals)
    cut=physical(stock,split_plan(offset=9.5),kerf=.2)
    check=assess(cut)
    assert len(cut["pieces"])==2
    tip=cut["pieces"]["rough_piece_3"]
    assert tip["usability"]["usable"]
    assert tip["usability"]["morphology"]=="pointed"
    assert check["usable_preform_weight_ct"]>=tip["weight_ct"]
    assert plan_score(cut,check)>plan_score(original,original_check)
    assert cut["retained"]<original["retained"]
    assert cut["balance"]["mass_balance_valid"]
    cut["plan"]["diagnostics"]["exact_sequence_verified"]=False
    assert assess(cut)["usable_preform_weight_ct"]==0


def test_usable_uncut_physical_stock_counts_in_full_even_with_source_metadata():
    stock=cells()
    stock.source_labels=np.zeros(len(stock.active),int)
    result=physical(stock)
    assert len(result["pieces"])==1
    assert assess(result)["usable_preform_weight_ct"]==100


def test_no_separation_status_cannot_authorize_a_nonempty_cut_sequence():
    result=physical(cells((10,5,5)),split_plan(offset=4.5),kerf=.2)
    result["plan"]["status"]="no_separation_required"
    assert assess(result)["usable_preform_weight_ct"]==0
