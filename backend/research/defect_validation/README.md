# Independent Defect Validation

This folder contains reproducibility helpers for the frozen independent
defect-detection final test. The workflow is independent of model selection and
does not retrain, tune thresholds, or alter labels.

## Frozen Evidence

- Test role: independent held-out final test
- Images: 45
- Specimen groups: 9
- Membership SHA-256:
  `4633d2e1fd7ce7772d06d8994c084e52f17b8cce1647587693d1558bc050e8cf`
- Label manifest SHA-256:
  `40d3d1cc061924c25192abaddffd6297353dc7afd0bd7e18f8de798436231e07`
- Selected model SHA-256:
  `5b7a362ed4c6670395e4693ab1a68d415ff1fb2164b2471cef37913338130b98`

Manual labels were frozen before inference. The final-test labels are not used
for model selection or threshold tuning.

## Frozen Evaluation Configuration

- Model type: Ultralytics YOLOv8n-seg
- Inference image size: 960
- Binary confidence threshold: 0.25
- Prediction NMS IoU: 0.7
- Native Ultralytics validation IoU: 0.7
- Native validation confidence: Ultralytics default
- Device: CPU

CPU is explicit because the current PyTorch CUDA build does not support the RTX
5050 `sm_120` target. CPU execution preserves the frozen inference and metric
configuration.

## Final Held-Out Result

- Precision: 0.03013444598980065
- Recall: 0.0009601163956491956
- F1: 0.0018609411709972319
- IoU: 0.0009313371673380055
- Native mask mAP50: 0.00043102434676857947
- Proposal target: 0.90
- Proposal target status: NOT ACHIEVED

Post-evaluation integrity audit: PASS. No evaluator defect was found. The poor
held-out result is valid final-test evidence and must be retained without rerun
or threshold tuning.

## Commands

Readiness before final inference:

```powershell
python backend\research\defect_validation\evaluate_independent_final_test.py --check-only
```

One-shot final evaluation:

```powershell
python backend\research\defect_validation\evaluate_independent_final_test.py --run-once
```

Post-evaluation integrity audit without inference:

```powershell
python backend\research\defect_validation\evaluate_independent_final_test.py --audit-result
```
