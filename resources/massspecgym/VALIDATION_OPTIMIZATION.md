# MiraFrag Validation Optimization Notes

This document tracks MassSpecGym validation/test experiments after correcting the evaluation protocol to exclude precursor peaks from scoring. Older precursor-including numbers are not comparable and should be treated as inflated.

The 2026-08-24 MSnLib experiment changed the interpretation of the whole validation program: direct metric fine-tuning (`sqrt_cosine` on MSnLib, `cosine` on MassSpecGym) can greatly improve scorer quality after KL/OOS pretraining. On MSnLib, the robust fine-tune reached test `sqrt_cosine_mean=0.74251`, far above FIORA's local precursor-excluded test `0.57416`. On MassSpecGym, direct `LOSS=cosine` fine-tuning from `checkpoints/mirafrag_unimol_retrieval_calibration.pt` reached `val_cosine=0.52863`, a modest but real gain over the retrieval checkpoint (`0.52484`). Treat older architecture experiments as negative under KL/OOS supervision, not as definitive under metric-aligned supervision.

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

The best current single-model MassSpecGym direction is still the action-primary checkpoint family around `0.5296`, but the fragment-only metric tests show that objective-only fine-tuning from that point is not enough. The earlier action-primary/physical-feature branch remains a useful architecture experiment around `0.5296`, but the MSnLib result shows that objective alignment can dominate architecture changes. Current cheap physical bond features do not add measurable gain beyond the action-primary branch, simple CE gating is noise-level, molecule descriptors regress, low-CE and rowwise-oracle distillation do not transfer posthoc complementarity, group-balanced row loss hurts the reported row-weighted metric, target-neighbor smoothing hurts row-level validation, and richer candidate support hurts unless the scorer is redesigned to use it.

The strongest practical MassSpecGym result is still the corrected UniMol/AIMNet ensemble around `0.543` test cosine, but the single-model path should first finish direct `cosine` and `sqrt_cosine` fine-tune evaluations before adding more architecture. If metric-aligned fine-tuning transfers into dense test exports, previous KL-trained architecture branches should only be revisited when they can be initialized as exact no-ops from the stronger metric-aligned checkpoint.
