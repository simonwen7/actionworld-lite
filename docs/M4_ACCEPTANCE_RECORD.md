# ActionWorld Lite — M4 Final Acceptance Record

**Milestone:** M4 — Action-conditioned Visual Physical World Model (PushT scene)  
**Record date:** 2026-10-26  
**Status:** **ACCEPTED — predeclared single-step performance and immutable TEST evidence checks PASSED**  
**Repository delivery:** Documentation and user-controlled Git commit/push remain pending; this status is distinct from scientific acceptance.  
**Next milestone:** M5 **NOT STARTED**.

## 1. Executive acceptance decision

The **IC-01** integrated PushT predictor (`IntegratedPushTWorldModel`) met **all four predeclared comparative criteria** on its **first and only held-out TEST execution** (Task 3AH), covering **1,701 single-step windows**. The archived one-time run exited `0`; Task 3AI-R1 independently verified all eight archived record hashes, all 19 frozen source/model/data hashes and the evaluator hash, the archived JSON cohort/count/metric consistency, and the four pass flags. No TEST examples were reopened in the forensic audit.

**Final single-step TEST headline metrics:** current XY mean **0.263062 cm**, next XY mean **0.339025 cm**, motion/displacement error mean **0.168006 cm**, current yaw circular MAE **2.179489°**, next yaw circular MAE **2.478032°**; next-contact precision **0.949260**, recall **0.967672**, F1 **0.958378**, ROC-AUC **0.998080**, and average precision **0.994938**.

**Scope of acceptance:** one-step predictions in the frozen synthetic MuJoCo PushT distribution. This is **not** a multi-step free-running rollout, an unseen-object/domain-generalization benchmark, or a claim of a general-purpose end-to-end world model.

## 2. Dataset and anti-leakage protocol

- Frozen scene-specific M3 data: **180 episodes**, 64 commanded control steps per episode; **126 TRAIN / 27 VALIDATION / 27 TEST** episodes; **7,938 / 1,701 / 1,701** overlapping one-step windows.
- Input: `RGB(t−1)` and `RGB(t)` concatenated into float32 `[B,6,64,64]` scaled to `[0,1]`, plus a float32 `[B,2]` action normalized by **0.02 m**. No future RGB is an input.
- Outputs: visually estimated current object XY/yaw, contact-gated displacement, predicted next XY, predicted next yaw, and final-step next-contact classification/score.
- At inference the public `forward(frames, action)` reads only these inputs, frozen parameters and fixed calibration/geometry constants; the true current/future simulator pose and labels are accessed only by the evaluator for scoring.
- **Contact label meaning:** pusher–T-block MuJoCo collision contact at the **end** of the commanded 0.05 s step (10 physics substeps); it is **not** the union of contacts over all substeps.
- Model, calibration, threshold and evaluator were fixed **before TEST**. TEST was opened **exactly once** by the Task 3AH evaluator. Test results were not used for training, model/threshold choice or modification. **TEST must not be rerun.**

## 3. Frozen integrated model (IC-01)

- **Runtime implementation:** `src/models/integrated_pusht_m4.py`; SHA-256 `e6aeb125e4556ec0855cc77a528754c1655bcb2f1f0d71ff2b1a140bff5fc49a` (**8,670 bytes**).
- **V0.3 dynamics**: frozen `GatedVisualPhysicalWorldModel`, **424,345 parameters**; produces action-conditioned XY and yaw increments with its learned motion gate.
- **V0.4 visual yaw**: frozen `SilhouetteYawSpecialist`, **231,362 parameters**, operating on the current image's centered orange silhouette.
- **Current block XY**: train-fitted fixed five-feature, 5×2 linear map of orange centroid and inferred yaw (cos, sin).
- **Pusher XY**: train-fitted fixed three-feature, 3×2 linear map of weak-blue centroid; observed detection covered 100% of the 1,701 validation windows in the earlier Task 3AB audit.
- **Next contact**: static endpoint disk-vs-T signed-clearance proxy (visually estimated current block held fixed). Predict contact iff `score_m >= 0.0008668007749070838`, with score `-clearance_m`. This threshold was selected once by maximizing TRAIN F1 in Task 3AC.
- **Translation hard gate**: predicted XY displacement is V0.3's already motion-gated XY delta if predicted contact is true, otherwise zero. Next XY = inferred current XY + this final displacement. **Yaw delta is not hard contact-gated**; next yaw rotates the V0.4 current yaw by V0.3's yaw delta.
- This is an **environment-specific hybrid visual/geometric/neural predictor**, not a newly trained end-to-end V0.5 network.

## 4. Final held-out TEST — headline and cohorts

All mean errors below are evaluated **once** on the sealed TEST, without any changes after observation.

| Cohort | Windows | Mean current XY (cm) | Mean next XY (cm) | Mean motion error (cm) | Mean current yaw (°) | Mean next yaw (°) |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 1,701 | 0.263062 | **0.339025** | **0.168006** | 2.179489 | **2.478032** |
| Center push | 567 | 0.290601 | 0.441184 | 0.303407 | 2.130387 | 2.484000 |
| Offset push | 567 | 0.263642 | 0.340948 | 0.200609 | 1.668601 | 2.224535 |
| Free motion | 567 | 0.234943 | 0.234942 | 0.000001 | 2.739479 | 2.725562 |
| Moving | 461 | 0.309331 | 0.577212 | 0.589360 | 1.903592 | 2.963984 |
| Static | 1,240 | 0.245860 | 0.250473 | 0.011357 | 2.282061 | 2.297368 |
| Contact | 464 | 0.304106 | 0.569696 | 0.580299 | 1.899886 | 2.953372 |
| No contact | 1,237 | 0.247666 | 0.252500 | 0.013354 | 2.284368 | 2.299732 |

- Max next-XY error: **2.342791 cm**; max next-yaw circular error: **20.807536°**.
- Overall next-XY mean / median / P90 / maximum: `0.339025 / 0.279380 / 0.677338 / 2.342791` cm.
- Overall next-yaw mean / median / P90 / maximum: `2.478032 / 2.142899 / 4.681698 / 20.807536` degrees.
- Overall motion-error mean / median / P90 / maximum: `0.168006 / ~0 / 0.676819 / 1.300830` cm (median at full precision about `6.887e-08` cm).

### Next-contact classification on the TEST

| Cohort | TP | FP | TN | FN | Precision | Recall | F1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | **449** | **24** | **1,213** | **15** | **0.949260** | **0.967672** | **0.958378** | **0.998080** | **0.994938** |
| Center push | 266 | 13 | 278 | 10 | 0.953405 | 0.963768 | 0.958559 | 0.994833 | 0.994514 |
| Offset push | 183 | 11 | 368 | 5 | 0.943299 | 0.973404 | 0.958115 | 0.997993 | 0.996023 |
| Free motion | 0 | 0 | 567 | 0 | — | — | — | — | — |
| Moving | 432 | 9 | 13 | 7 | 0.979592 | 0.984055 | 0.981818 | 0.976289 | 0.998763 |
| Static | 17 | 15 | 1,200 | 8 | 0.531250 | 0.680000 | 0.596491 | 0.990189 | 0.613572 |

A dash indicates an undefined single-class metric, **not zero**. The Contact-only and No-contact-only cohorts are also single-class for ranking metrics.

## 5. Four predeclared TEST comparison gates

| Criterion, recomputed from TEST baselines | IC-01 (cm) | TEST baseline (cm) | Result |
|---|---:|---:|---|
| Overall mean motion error **<** zero-motion mean | **0.168006** | 0.254862 | **PASS** |
| Moving mean motion error **≤** native V0.2 raw moving | **0.589360** | 0.700333 | **PASS** |
| Static mean motion error **≤** native V0.2 raw static | **0.011357** | 0.154465 | **PASS** |
| Overall mean re-anchored next XY error **≤** V0.2 re-anchored next XY | **0.339025** | 0.390259 | **PASS** |

**All four predeclared comparisons = TRUE.** The comparator figures are the corresponding **TEST** baselines, not Validation numbers. Native V0.2/0.3 TEST next-XY errors were 2.823264/2.783739 cm before visual re-anchoring. The zero-motion overall next-XY error with visual re-anchoring was 0.431118 cm.

## 6. Validation-to-TEST context (not a selection procedure)

| IC-01 metric | VALIDATION | Held-out TEST |
|---|---:|---:|
| Mean current XY error (cm) | 0.336028 | 0.263062 |
| Mean next XY error (cm) | 0.410735 | 0.339025 |
| Mean motion error (cm) | 0.156625 | 0.168006 |
| Mean current yaw error (°) | 1.627133 | 2.179489 |
| Mean next yaw error (°) | 1.865655 | 2.478032 |
| Contact F1 | 0.951194 | 0.958378 |

The validation and training data were used earlier for research, model comparison and train-only fitting respectively. This post-hoc table **must not** be used to revise IC-01 or repeat TEST.

## 7. Scope limits, risks and honest interpretation

1. **Single-step only:** the model has not been validated when repeatedly consuming its own predicted future state; multi-step rollout is deferred to the separately scoped next milestone.
2. **Specialized scene prior:** segmentation depends on orange T-block / blue pusher colors, a particular camera and scale, fixed T geometry, and constrained PushT actions. No claim of transfer to arbitrary objects, colors, cameras or environments.
3. **Static-contact minority:** the held-out Static cohort includes only 25 positive contact labels (`TP=17, FN=8`). Its contact F1 is **0.596491** despite much higher overall F1, and its false positives are 15. It deserves explicit presentation.
4. **Motion residuals:** moving-window displacement error remains **0.589360 cm mean**, with a moving next-XY mean **0.577212 cm**; strong aggregate averages do not mean physically perfect state transitions.
5. **Contact semantics:** target is final-substep contact, not any transient collision during the action. The clearance proxy holds the block fixed and is not an actual 10-substep physics solver.
6. **No future observation input or ground-truth pose at runtime:** the inference API is image/action-only; true state and label data are held by the evaluator solely to score predictions.
7. **Test integrity:** future retuning or a second evaluation on this same TEST would compromise its independent held-out role. Preserve and cite the original archived run instead.

## 8. Evidence provenance and immutable artifacts

**Original unique TEST execution:** Task 3AH, exit status `0`, Task 3AG evaluator `--run-one-time-test`, CPU, 1,701 windows. Executed **once**. Archived at:

`/Users/arrebol/Desktop/ActionWorldLite_M4_Task3AH_ONE_TIME_HELDOUT/`

**Archive hashes** (read-only Task 3AI-R1 independently verified all eight against original Task 3AH):

| Archived file | SHA-256 |
|---|---|
| `run_manifest.txt` | `d850064f59a4a1fdbb753c68ef5b52eae52c2eb40ea34a5127d59e1aa59a6f96` |
| `one_time_execution_status.txt` | `08fb4e683bb0809aac8b3e1c32d807c78124640c4963395c11741150d4bf2b40` |
| `evaluator_stdout.log` | `b930d69d95006684ad727afab55013e1f666b23129f129ddf927fde667acf093` |
| `evaluator_stderr.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `post_integrity.log` | `5e70306806b7bc0f4d386ceb26ce05d1043f1b667167d9ed4bfb8185e1df048d` |
| `git_status_after.txt` | `27d8a0a4c1e953ab1f95e240689ccbfb905c453352a917c1bb34f3c8f54ba84d` |
| `git_diff_check.txt` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `report_summary.txt` | `077fe7a1072e5c94014e59e8c854bca6227953fbea3d8a5f3c892aac7cf21f73` |

**Frozen integrated model SHA-256:** `e6aeb125e4556ec0855cc77a528754c1655bcb2f1f0d71ff2b1a140bff5fc49a`  
**Frozen TEST evaluator:** `src/eval/evaluate_m4_heldout.py`, **20,381 bytes**, SHA-256 `3d9d87009440a646bbcdcfae488a47c0394d7d299a84b72f1ac38e6476f257b5`.

**Task 3AF:** 19 source/model/data hashes matched; source audit found no privileged inference leakage.  
**Task 3AG:** pre-registered evaluator synthetic metric QA passed without opening TRAIN, VALIDATION or TEST.  
**Task 3AH:** one-time held-out TEST passed; evaluator and frozen hashes verified before and after, stderr empty.  
**Task 3AI:** first forensic audit stopped on a 63-character *expected* manifest digest typo, before checking original records; **no archive changes occurred**.  
**Task 3AI-R1:** corrected only that missing hexadecimal character; all 8 archive hashes and all 19 frozen hashes plus evaluator verified; archived JSON and all four comparisons verified; **no dataset opened, no inference, no TEST rerun, no file writes**.

The full original, immutable TEST JSON is stored in `evaluator_stdout.log`; its hash above is the traceable reference. The record here is a human-readable synopsis, not a replacement for original per-cohort data.

## 9. Repository handoff and milestone boundary

At completion of Task 3AI-R1, the working tree had **10 untracked M4 sources**, intentionally not committed or pushed by Cursor:

```text
src/data/world_model_dataset.py
src/eval/evaluate_m4_heldout.py
src/models/integrated_pusht_m4.py
src/models/visual_dynamics.py
src/models/visual_dynamics_v03.py
src/models/yaw_specialist_v04.py
src/train/train_visual_dynamics.py
src/train/train_visual_dynamics_v02.py
src/train/train_visual_dynamics_v03.py
src/train/train_yaw_specialist_v04.py
```

- M4 **scientific/performance acceptance: PASSED**; M4 **archived-evidence integrity: PASSED**.
- This Markdown acceptance record is the final documentation handoff; adding it to the repository does **not** authorize a new experiment or any TEST access.
- The user alone decides when to `git add`, `git commit`, and `git push`; Cursor must not perform Git writes.
- The frozen TEST remains consumed. **Do not run `--run-one-time-test` again**.
- **M5 work is not authorized and has not started.**

**Final decision:** **ACCEPT M4 IC-01 for the defined single-step PushT world-model benchmark**, subject only to user-controlled repository documentation and Git archival. Do not equate this acceptance with verified multi-step simulation performance.
