# MiraFrag Validation Optimization Notes

This document tracks MassSpecGym validation/test experiments after correcting the evaluation protocol to exclude precursor peaks from scoring. Older precursor-including numbers are not comparable and should be treated as inflated.

For clean MassSpecGym comparisons, keep three categories separate:

- clean single-model runs trained from the standard pretrained molecular encoder without a task checkpoint;
- single-model runs initialized from earlier MassSpecGym checkpoints;
- posthoc ensembles, which are useful diagnostics but not the target deployment shape.

The best clean single-model MassSpecGym numbers before the current action-geometry scratch rerun were the action-primary checkpoint family around `0.5296` validation cosine. The strongest posthoc ensemble reached about `0.543` corrected test cosine, but should be reported separately from single-model results.

### Direct cosine fine-tuning, 2026-08-24

Initialized from `checkpoints/mirafrag_unimol_retrieval_calibration.pt` and trained with `LOSS=cosine`, `LR=1e-5`, `ENCODER_LR=3e-6`, `WEIGHT_DECAY=0`, `HEAD_WEIGHT_DECAY=0`, and `EXPONENTIAL_GAMMA=0.98`.

Observed validation trajectory:

- epoch 1: `val_cosine=0.52482`, `val_oos=0.04301`
- epoch 3: `val_cosine=0.52749`, `val_oos=0.01251`
- epoch 5: `val_cosine=0.52833`, `val_oos=0.00429`
- epoch 9: best observed `val_cosine=0.52863`, `val_oos=0.00053`
- epoch 12: `val_cosine=0.52752`, `val_oos=0.00014`

Interpretation: objective mismatch is present on MassSpecGym, but less dominant than on MSnLib. The direct cosine objective suppresses OOS and improves emitted-spectrum scoring by about `+0.0038` validation cosine. However, the later fragment-only objective tests show that much of the direct-cosine gain came from OOS renormalization rather than a better fragment scorer. Dense exported val/test evaluation of `mirafrag_unimol_cosine_ft.pt` is still useful, but objective-only fine-tuning no longer looks like the main route to a large single-model improvement.

### Fragment-only metric fine-tuning, 2026-08-24

To separate fragment scoring from OOS renormalization, two new losses were added: `fragment_cosine` and `fragment_sqrt_cosine`. Both use fragment-only softmax probabilities and ignore the OOS head. Checkpoints trained with these losses are exported with decoupled fragment-softmax semantics.

Starting from `checkpoints/mirafrag_action_primary_ablation.pt`, `LOSS=fragment_cosine`, `LR=1e-5`, `ENCODER_LR=3e-6`, `WEIGHT_DECAY=0`, `HEAD_WEIGHT_DECAY=0`, and `EXPONENTIAL_GAMMA=0.98` produced:

- epoch 0: `val_fragment_cosine_loss=0.47035`, `val_cosine=0.52965`, `val_oos=0.25126`
- epoch 1: `val_cosine=0.52817`, `val_oos=0.25129`
- epoch 2: `val_cosine=0.52810`, `val_oos=0.25133`
- epoch 3: `val_cosine=0.52782`, `val_oos=0.25136`

Result: negative but diagnostic. OOS no longer collapses, proving the objective behaves as intended, but validation decreases immediately. The action-primary checkpoint is already near a local optimum for fragment-only cosine.

The matched `LOSS=fragment_sqrt_cosine` run produced:

- epoch 0: `val_fragment_sqrt_cosine_loss=0.46250`, `val_cosine=0.52965`, `val_oos=0.25126`
- epoch 1: `val_fragment_sqrt_cosine_loss=0.45856`, `val_cosine=0.51865`, `val_oos=0.25137`
- epoch 2: `val_fragment_sqrt_cosine_loss=0.45997`, `val_cosine=0.51726`, `val_oos=0.25140`

Result: negative for the main `val_cosine` metric. The sqrt loss improves its own validation loss while sharply lowering ordinary cosine, so it changes peak weighting in the wrong direction for the current MassSpecGym target metric.

Conclusion: direct full `cosine` fine-tuning was partly useful because it suppressed OOS. Once OOS is removed from the objective, metric-only fragment fine-tuning overfits immediately. Do not continue objective-only fine-tuning from `mirafrag_action_primary_ablation.pt` unless checkpoint selection switches to a sqrt-weighted metric.

## Current Best Direction: Metric-Aligned UniMol Fine-Tuning

### Clean action-geometry v2 scratch run, 2026-09-05

This is the current reconstructable from-scratch MassSpecGym control after the cleanup. It starts from the standard Uni-Mol v1 pretrained encoder, with no MiraFrag task checkpoint initialization. It uses `FRAGMENT_PATH_LAYERS=3`, `FRAGMENT_ACTION_PRIMARY_LAYERS=2`, `BOND_BREAK_GEOMETRY_FEATURES=1`, and the v2 support-generation changes.

Observed validation trajectory:

- epoch 1: `val_cosine=0.47009`, `val_decoupled_kl=1.78616`, `val_oos=0.20473`
- epoch 3: `val_cosine=0.50545`, `val_decoupled_kl=1.65556`, `val_oos=0.22756`
- epoch 8: `val_cosine=0.51529`, `val_decoupled_kl=1.64547`, `val_oos=0.22246`
- epoch 12: `val_cosine=0.52305`, `val_decoupled_kl=1.61128`, `val_oos=0.21340`
- epoch 14: `val_cosine=0.52546`, `val_decoupled_kl=1.61206`, `val_oos=0.24531`
- epoch 32: `val_cosine=0.52990`, `val_decoupled_kl=1.62671`, `val_oos=0.22786`
- epoch 37: `val_cosine=0.53252`, `val_decoupled_kl=1.58916`, `val_oos=0.21916`
- epoch 52: best observed `val_cosine=0.53468`, `val_decoupled_kl=1.61569`, `val_oos=0.22879`
- epochs 53-64 stayed around `0.53025-0.53379`

Interpretation: this is a real improvement over the earlier clean action-geometry scratch run, which peaked around `0.5197`, and it now also beats the older action-primary checkpoint family around `0.5296`. The gain came from the model-side v2 changes alone, because this continuation did not enable the new direct bond-cut fragment support. The best checkpoint so far is epoch 52.

### Direct bond-cut fragment support, 2026-09-06

This run tests the new fragment-support expansion on top of the action-geometry v2 scratch setup. It enables direct connected components from up to two original bond cuts with `DIRECT_BOND_CUT_FRAGMENTS=1`, `MAX_DIRECT_BOND_CUTS=2`, `MAX_FRAGMENTS=2048`, and `MAX_FRAGMENT_EDGES=8192`.

Observed validation trajectory so far:

- epoch 1: `val_cosine=0.46822`, `val_decoupled_kl=1.97674`, `val_oos=0.17512`
- epoch 5: `val_cosine=0.51470`, `val_decoupled_kl=1.76425`, `val_oos=0.19867`
- epoch 12: `val_cosine=0.52470`, `val_decoupled_kl=1.78014`, `val_oos=0.18107`
- epoch 21: `val_cosine=0.52988`, `val_decoupled_kl=1.75962`, `val_oos=0.18393`
- epoch 22: `val_cosine=0.53241`, `val_decoupled_kl=1.75158`, `val_oos=0.18021`
- epoch 25: `val_cosine=0.53434`, `val_decoupled_kl=1.75902`, `val_oos=0.19326`
- epoch 31: `val_cosine=0.53455`, `val_decoupled_kl=1.77563`, `val_oos=0.18779`
- epoch 33: `val_cosine=0.53482`, `val_decoupled_kl=1.76641`, `val_oos=0.18687`
- epoch 37: best observed so far `val_cosine=0.53890`, `val_decoupled_kl=1.74995`, `val_oos=0.19025`
- epoch 38: `val_cosine=0.53484`, `val_decoupled_kl=1.78530`, `val_oos=0.19627`

Full validation eval of the saved direct-cut checkpoint reproduced the training-loop score: `n=9734`, `cosine_mean=0.53890`, `sqrt_cosine_mean=0.54412`, `candidate_coverage_mean=0.79364`, `oos_target_mass_mean=0.20636`, `oracle_binned_cosine_mean=0.89187`, `oracle_tolerance_cosine_mean=0.91630`, `support_gap_mean=0.10813`, `scorer_gap_mean=0.35297`, and `oos_calibration_abs_error_mean=0.16164`.

Stratified validation diagnostics show that the remaining error is not uniform. Orbitrap rows are much stronger than QTOF (`0.56037` vs. `0.42313` cosine). Collision energies above 60 are the weakest CE bin (`0.36692` cosine, `0.59535` coverage), while low/mid CE bins are around `0.539-0.565`. The hardest subgroup is QTOF above 60 CE (`n=222`, `cosine=0.32737`, `coverage=0.54388`); Orbitrap at 20-30 CE is the strongest subgroup (`n=1361`, `cosine=0.58231`, `coverage=0.87840`).

Interpretation: positive. Direct bond-cut support now beats the model-only v2 scratch baseline of `0.53468` at epoch 52, reaching `0.53890` already by epoch 37 and reproducing this value in full validation eval. The lower OOS values around `0.18-0.21`, plus validation `candidate_coverage_mean=0.79364`, suggest the expanded support is capturing more target mass rather than merely reshuffling logits inside the old candidate set. This is the new best clean single-model MassSpecGym validation result recorded here.

### Three-direct-cut fragment support, 2026-09-07/08

The corrected `MAX_DIRECT_BOND_CUTS=3` oracle used matched base and high-CE candidate budgets (`MAX_FRAGMENTS=4096`, `MAX_FRAGMENT_EDGES=16384`, `HIGH_CE_MAX_FRAGMENTS=4096`, `HIGH_CE_MAX_FRAGMENT_EDGES=16384`). This avoids the earlier accidental high-CE budget reduction.

Oracle result versus the cuts=2 best support:

- overall coverage: `0.79364 -> 0.82527` (`+0.03163`)
- overall tolerance oracle: `0.91630 -> 0.93532` (`+0.01902`)
- CE > 60 coverage: `0.59535 -> 0.67810` (`+0.08275`)
- CE > 60 tolerance oracle: `0.79905 -> 0.86157` (`+0.06253`)
- QTOF coverage: `0.61773 -> 0.65161` (`+0.03388`)
- QTOF tolerance oracle: `0.84277 -> 0.86536` (`+0.02259`)

Training from scratch with cuts=3 required reducing the batch size to 16 because `BATCH_SIZE=48` OOMed in the action-primary pair scorer. The run uses `FRAGMENT_ACTION_PRIMARY_LAYERS=2`, geometry/local-environment/CE-conditioned action scoring, `LR=1e-4`, `ENCODER_LR=3e-5`, and `EXPONENTIAL_GAMMA=0.97`.

Observed validation trajectory so far:

- epoch 4: `val_cosine=0.52539`, `val_decoupled_kl=1.84648`, `val_oos=0.15057`
- epoch 8: `val_cosine=0.53398`, `val_decoupled_kl=1.85334`, `val_oos=0.15559`
- epoch 10: `val_cosine=0.53557`, `val_decoupled_kl=1.87329`, `val_oos=0.16279`
- epoch 11: `val_cosine=0.53924`, `val_decoupled_kl=1.81825`, `val_oos=0.15187`
- epoch 17: `val_cosine=0.53987`, `val_decoupled_kl=1.88402`, `val_oos=0.15876`
- epoch 25: `val_cosine=0.54022`, `val_decoupled_kl=1.91552`, `val_oos=0.15801`
- epoch 27: `val_cosine=0.54422`, `val_decoupled_kl=1.89650`, `val_oos=0.15410`
- epoch 33: `val_cosine=0.54479`, `val_decoupled_kl=1.89898`, `val_oos=0.15816`
- epoch 34: `val_cosine=0.54124`, `val_decoupled_kl=1.90767`, `val_oos=0.16030`
- epoch 37: best observed so far `val_cosine=0.54487`, `val_decoupled_kl=1.93081`, `val_oos=0.15703`
- epochs 38-45: stayed in `0.54103-0.54437`, with no new best

Interpretation: positive, likely near plateau. Cuts=3 now clearly beats the cuts=2 direct-cut best of `0.53890`, reaching `0.54487` by epoch 37. The validation OOS is lower (`~0.15-0.16`) than the cuts=2 run (`~0.18-0.20`), consistent with the higher oracle support coverage. This is now the best clean single-model MassSpecGym validation result recorded here. Further epochs after 37 show only noise-level movement, so continuing much longer is low priority unless compute is otherwise idle.

### Regularized continuation from cuts=3 best, 2026-09-09/10

Initialized from `mirafrag_action_geometry_v2_direct_cuts3_scratch_bs16.pt` and kept the cuts=3 support/cache. This tests whether weight decay can stabilize the best cuts=3 solution, after the unregularized low-LR continuation had regressed. Settings were `LR=1e-5`, `ENCODER_LR=3e-6`, `WEIGHT_DECAY=3e-3`, `HEAD_WEIGHT_DECAY=1e-4`, and `EXPONENTIAL_GAMMA=0.98`.

Observed validation trajectory:

- epoch 0: `val_cosine=0.54487`, `val_decoupled_kl=1.93081`, `val_oos=0.15703`
- epoch 1: `val_cosine=0.54503`, `val_decoupled_kl=1.92448`, `val_oos=0.15554`
- epoch 3: `val_cosine=0.54547`, `val_decoupled_kl=1.91998`, `val_oos=0.15828`
- epoch 8: `val_cosine=0.54503`, `val_decoupled_kl=1.93413`, `val_oos=0.15870`
- epoch 12: best observed so far `val_cosine=0.54618`, `val_decoupled_kl=1.91761`, `val_oos=0.15611`
- epochs 13-15: stayed below the best, `0.54408-0.54481`

Full validation eval reproduced the training-loop score: `n=9734`, `cosine_mean=0.54618`, `sqrt_cosine_mean=0.55457`, `candidate_coverage_mean=0.82517`, `oos_target_mass_mean=0.17483`, `oracle_binned_cosine_mean=0.91006`, `oracle_tolerance_cosine_mean=0.93521`, `support_gap_mean=0.08994`, `scorer_gap_mean=0.36388`, and `oos_calibration_abs_error_mean=0.14437`.

Stratified validation diagnostics for this checkpoint show the remaining error is still concentrated in instrument/CE subgroups:

- instrument: QTOF `cosine=0.41897`, `coverage=0.65068`, `oracle_tolerance=0.86464`; Orbitrap `cosine=0.56977`, `coverage=0.85754`, `oracle_tolerance=0.94830`
- CE > 60: `cosine=0.38301`, `coverage=0.67759`, `oracle_tolerance=0.86157`
- CE 30-60: `cosine=0.54688`, `coverage=0.83666`, `oracle_tolerance=0.94376`
- CE 20-30: `cosine=0.57114`, `coverage=0.87664`, `oracle_tolerance=0.95858`
- CE <= 20: `cosine=0.57032`, `coverage=0.82090`, `oracle_tolerance=0.93082`
- hardest subgroup: QTOF and CE > 60, `n=222`, `cosine=0.31754`, `coverage=0.61911`, `oracle_tolerance=0.88121`
- strongest subgroup: Orbitrap and CE <= 20, `n=2782`, `cosine=0.60516`, `coverage=0.86916`, `oracle_tolerance=0.95491`

Interpretation: small positive and now independently confirmed. Unlike the unregularized continuation, this regularized continuation improved the checkpoint from `0.54487` to `0.54618`. The full eval shows the support gap is now down to about `0.09`, while scorer gap remains large at about `0.36`; future model changes should therefore focus more on ranking/calibrating the expanded candidate set, especially for QTOF/high-CE rows, than on broad support expansion alone. Later epochs dipped, so this should be treated as a short stabilization/refinement result rather than evidence for long continuation.

### Low-LR continuation from direct-cut best, 2026-09-06

Initialized from the best direct-cut scratch checkpoint and reused the same direct-cut support/cache. Settings were `LR=2e-5`, `ENCODER_LR=6e-6`, `WEIGHT_DECAY=0`, `HEAD_WEIGHT_DECAY=0`, and `EXPONENTIAL_GAMMA=0.98`.

Observed validation trajectory:

- epoch 0: `val_cosine=0.53890`, `val_decoupled_kl=1.74995`, `val_oos=0.19025`
- epoch 1: `val_cosine=0.53605`, `val_decoupled_kl=1.78001`, `val_oos=0.18472`
- epoch 2: `val_cosine=0.53578`, `val_decoupled_kl=1.77868`, `val_oos=0.18202`
- epoch 4: `val_cosine=0.53637`, `val_decoupled_kl=1.78235`, `val_oos=0.19185`
- epoch 7: `val_cosine=0.53469`, `val_decoupled_kl=1.80064`, `val_oos=0.18907`
- epoch 9: `val_cosine=0.53460`, `val_decoupled_kl=1.78883`, `val_oos=0.19513`

Interpretation: negative. The best value is the epoch-0 checkpoint copy; every trained epoch is worse. Continuing the same model at lower learning rate moves away from the current best, so further improvement likely needs a support/model change rather than more fine-tuning of this checkpoint.

### Physical bond sidecar + action-primary branch, 2026-08-19

Command family: initialized from `checkpoints/mirafrag_unimol_retrieval_calibration.pt`, enabled `FRAGMENT_ACTION_PRIMARY_LAYERS=1`, low learning rates (`LR=1.5e-5`, `ENCODER_LR=5e-6`, `EXPONENTIAL_GAMMA=0.97`), and trained with the existing row-level MassSpecGym cache. The physical sidecar run additionally used `PHYSICAL_BOND_FEATURES=1` and `data/physical_bond_features/physical_features.csv`.

Best validation for the physical-sidecar run:

- epoch 6: `val_cosine=0.52962`, `val_decoupled_kl=1.61671`, `val_oos=0.25607`
- test evaluation: `cosine_mean=0.52947`, `sqrt_cosine_mean=0.53448`, `candidate_coverage_mean=0.74436`, `oos_target_mass_mean=0.25564`, `oracle_binned_cosine_mean=0.86189`, `oracle_tolerance_cosine_mean=0.89051`, `support_gap_mean=0.13811`, `scorer_gap_mean=0.33242`, `oos_calibration_abs_error_mean=0.19538`

### Action-primary ablation without physical sidecar, 2026-08-19

Same setup but without `PHYSICAL_BOND_FEATURES`.

Best validation:

- epoch 1: `val_cosine=0.52965`, `val_decoupled_kl=1.61954`, `val_oos=0.25126`
- later epochs stayed around `0.5266-0.5283`

Interpretation: the observed validation improvement is caused by the action-primary branch and fine-tuning setup, not by the current RDKit physical sidecar columns. The physical sidecar is not harmful, but it is not the driver of the gain in this ablation.

### CE-gated action-primary branch, 2026-08-19

Initialized from `checkpoints/mirafrag_action_primary_ablation.pt` and enabled `FRAGMENT_ACTION_PRIMARY_CE_GATE=1`. The CE gate is zero-initialized, so epoch 0 reproduced the ablation checkpoint.

Best validation:

- epoch 0: `val_cosine=0.52965`, `val_decoupled_kl=1.61954`, `val_oos=0.25126`
- epoch 1: `val_cosine=0.52968`, `val_decoupled_kl=1.60749`, `val_oos=0.25584`
- epochs 2-4 regressed to `0.52625-0.52850`

Interpretation: explicit CE gating on the action-primary evidence did not materially improve validation. The `+0.00003` gain is within noise, so the action-primary architecture remains useful, but this simple CE-gate variant should not be treated as a real improvement.


### Target-neighbor smoothing, 2026-08-19/20

Initialized from `checkpoints/mirafrag_action_primary_ablation.pt` and enabled train-only target smoothing with same-SMILES, same-instrument, same-adduct neighbors within a raw collision-energy window of 10 eV. Neighbor coverage was high: `rows_with_neighbors=85004/99341` at `weight=0.3`.

Validation result:

- epoch 0: `val_cosine=0.52965`, `val_decoupled_kl=1.68049`, `val_oos=0.25126`
- epoch 1: `val_cosine=0.52191`, `train_cosine=0.85380`, `val_oos=0.22451`
- epoch 2: `val_cosine=0.52037`, `train_cosine=0.85437`, `val_oos=0.22422`

Interpretation: smoothing applies broadly but hurts row-level validation immediately. The model learns the smoothed/averaged target distribution, but the official objective scores each exact MassSpecGym row. This supports stopping target-neighbor smoothing for row-level validation, though it may still be useful for merged-spectrum objectives.

### Rich high-CE support on action-primary checkpoint, 2026-08-20

Initialized from `checkpoints/mirafrag_action_primary_ablation.pt`, kept `FRAGMENT_ACTION_PRIMARY_LAYERS=1`, and restored a richer fragment support set: base `depth=3`, `broken_bonds=6`, `max_fragments=2048`, `max_edges=8192`; high-CE threshold 60 with `depth=4`, `broken_bonds=8`, `max_fragments=4096`, `max_edges=16384`.

Validation result:

- epoch 0: `val_cosine=0.52300`, `val_decoupled_kl=1.89271`, `val_oos=0.25126`
- epoch 1: `val_cosine=0.52359`, `train_cosine=0.83149`, `val_oos=0.19070`
- epoch 3: best observed `val_cosine=0.52436`, `train_cosine=0.83907`, `val_oos=0.19332`
- epochs 4-11 stayed around `0.52078-0.52390`

Interpretation: richer support alone does not help this scorer. The lower epoch-0 cosine shows that the checkpoint's learned scores are calibrated to the smaller candidate universe. Adding many plausible candidates increases ambiguity faster than it adds useful matched support. Future support expansion needs a model change that learns to gate or rank the expanded candidates, not just a larger candidate list.

### Bond-event GNN residual on action-primary checkpoint, 2026-08-20

Initialized from `checkpoints/mirafrag_action_primary_ablation.pt` and enabled `FRAGMENT_ACTION_BOND_GNN_LAYERS=2`. This adds grouped message passing over broken-bond events inside each candidate formula, while preserving the loaded checkpoint as a zero-initialized residual.

Validation result:

- epoch 0: `val_cosine=0.52965`, `val_decoupled_kl=1.61954`, `val_oos=0.25126`
- epoch 1: `val_cosine=0.52744`, `train_cosine=0.83604`, `val_oos=0.25508`
- epoch 4: best after training `val_cosine=0.52865`, `train_cosine=0.83655`, `val_oos=0.25778`
- epochs 5-15 stayed around `0.52519-0.52829`

Interpretation: the implementation behaved correctly because epoch 0 exactly reproduced the action-primary checkpoint, but the additional bond-event interaction branch did not improve validation. This suggests that simply adding more local multi-break capacity on top of the current candidate scorer is not enough; the remaining gap is more likely in target/scoring calibration, expert selection, or the candidate-support/scorer tradeoff than in this specific local event interaction.

### Molecule descriptor branch, 2026-08-22

Initialized from `checkpoints/mirafrag_action_primary_ablation.pt` and enabled `--molecule-descriptor-features` with a 128-dim descriptor MLP.

Validation result:

- epoch 0: `val_cosine=0.52965`, `val_decoupled_kl=1.68049`, `val_oos=0.25126`
- epoch 1: `val_cosine=0.52314`, `train_cosine=0.83413`, `val_oos=0.23475`
- epoch 3: best trained point `val_cosine=0.52427`, `train_cosine=0.83481`, `val_oos=0.23743`
- epoch 5: `val_cosine=0.52159`, `train_cosine=0.83530`, `val_oos=0.23829`

Interpretation: generic molecule descriptors did not explain the subset-specific model differences. The branch was trainable, but it moved away from the strong initialization and did not improve row-level validation.

### Low-CE targeted AIMNet distillation, 2026-08-22

Plateau diagnostics showed a tempting non-ensemble opportunity in the low-collision-energy regime. A posthoc rule using AIMNet-style predictions for low CE and UniMol otherwise suggested that CE <= roughly 20-28 eV could be the weak subset for the UniMol checkpoint. We therefore tested train-time-only CE-gated distillation: apply AIMNet/high-support teacher spectra only to low-CE training rows using `--distill-ce-max`, while keeping a single UniMol model at inference.

The run initialized from `checkpoints/mirafrag_action_primary_ablation.pt`, loaded 36,827 low-CE teacher rows with `filter=ce_max=20`, used `DISTILL_LOSS_WEIGHT=0.02`, and kept the low fine-tuning rates (`LR=1.5e-5`, `ENCODER_LR=5e-6`, `EXPONENTIAL_GAMMA=0.97`).

Validation result:

- epoch 0: `val_cosine=0.52965`, `val_decoupled_kl=1.68049`, `val_oos=0.25126`
- epoch 1: `val_cosine=0.52287`, `train_distill_loss=0.56726`
- epoch 3: `val_cosine=0.52429`, `train_distill_loss=0.52408`
- epoch 9: `val_cosine=0.52460`, `train_distill_loss=0.49730`
- epoch 10: best trained point `val_cosine=0.52496`, `train_distill_loss=0.49467`

Interpretation: negative. Distillation lowered the teacher loss but did not transfer the posthoc CE-bin complementarity into one model. The best checkpoint remains the epoch-0 initialization. This suggests the AIMNet/UniMol complementarity is not captured by a simple low-CE teacher penalty.

### Rowwise oracle UniMol/AIMNet distillation, 2026-08-25

This tested whether the strong posthoc complementarity between UniMol and AIMNet could be folded into one model without using an inference-time ensemble. The teacher target was built from two train-set prediction files using rowwise no-precursor cosine against the true training spectrum. The run initialized from `checkpoints/mirafrag_action_primary_ablation.pt`, used `DISTILL_MODE=rowwise-oracle`, `DISTILL_LOSS_WEIGHT=0.02`, `DISTILL_ORACLE_TEMPERATURE=0.03`, and `DISTILL_ORACLE_MIN_TEACHER_WEIGHT=0.05`.

Teacher diagnostic:

```text
Rowwise-oracle distillation weights: rows=99341 first_wins=82471 second_wins=16870 first_weight_mean=0.6888 first_cosine_mean=0.84070 second_cosine_mean=0.76635 oracle_cosine_mean=0.84994 temperature=0.03 min_teacher_weight=0.05
```

Validation result:

- epoch 0: `val_cosine=0.52965`, `val_decoupled_kl=1.68049`, `val_oos=0.25126`
- epoch 1: `val_cosine=0.52182`, `train_distill_loss=0.15559`
- epoch 3: `val_cosine=0.52311`, `train_distill_loss=0.15384`
- epoch 9: best trained point `val_cosine=0.52353`, `train_distill_loss=0.15276`

Interpretation: negative. The rowwise teacher oracle is strong on the training rows, but projected teacher KL still moves validation away from the true row-level target. The result confirms that UniMol/AIMNet complementarity is real but not trivially distillable into the current single scorer. The best checkpoint remains the epoch-0 initialization.

### Group-balanced row loss, 2026-08-21

This tested whether overrepresented replicate groups were dominating the row-level objective. The run initialized from `checkpoints/mirafrag_action_primary_ablation.pt`, enabled `--group-balanced-loss --group-balance-cols auto --group-balance-power 1.0`, and produced weights over 15,516 groups:

```text
Group-balanced loss weights: rows=99341 groups=15516 cols=smiles,adduct,instrument_type power=1 min=0.02858 max=6.402 mean=1
```

Validation result:

- epoch 0: `val_cosine=0.52965`
- epoch 1: best trained point `val_cosine=0.52276`
- epochs 2-6 stayed around `0.52104-0.52268`

Interpretation: negative for the current row-level validation objective. Equalizing replicate groups hurts the row-weighted metric we actually report.

## Practical Conclusion

The best current clean single-model MassSpecGym result is now the regularized action-geometry v2 cuts=3 continuation at `val_cosine=0.54618` on continuation epoch 12. This beats the model-only v2 scratch run at `0.53468`, the earlier clean action-geometry scratch run around `0.5197`, and the older action-primary checkpoint family around `0.5296`. Current cheap physical bond features do not add measurable gain beyond the action-primary branch, simple CE gating is noise-level, molecule descriptors regress, low-CE and rowwise-oracle distillation do not transfer posthoc complementarity, group-balanced row loss hurts the reported row-weighted metric, and target-neighbor smoothing hurts row-level validation.

The strongest practical MassSpecGym result is still the corrected UniMol/AIMNet ensemble around `0.543` test cosine, but the single-model path should be reported separately. If new architecture branches are tested, they should either start from the same clean scratch recipe or initialize as exact no-ops from the `~0.5296` action-primary checkpoint family so the comparison is interpretable.
