# MSnLib Experiments

This file tracks the MSnLib/FIORA comparison using the converted MiraFrag TSV in `data/MSnLib_mirafrag.tsv`.

## Dataset

The converter builds a MiraFrag-compatible file from the FIORA MSnLib resources with fixed train/validation/test splits. The current split sizes are:

```text
train       110715
validation   13746
test         13683
```

These experiments use the same candidate-fragment machinery as the MassSpecGym work, but MSnLib exposed a much stronger objective mismatch: KL/OOS training produced reasonable train curves while underperforming badly under the exported square-root-cosine metric.

## FIORA Reference

Local FIORA evaluation on the same validation/test splits:

```text
validation n=13746
spectral_sqrt_cosine_mean=0.61923
spectral_sqrt_cosine_wo_prec_mean=0.57763
spectral_sqrt_cosine_avg_mean=0.59843

test n=13683
spectral_sqrt_cosine_mean=0.61689
spectral_sqrt_cosine_wo_prec_mean=0.57416
spectral_sqrt_cosine_avg_mean=0.59553
```

The precursor-excluded value is the stricter comparison. The default FIORA value is still useful as a sanity reference because it includes the easier precursor contribution.

## MiraFrag KL-Trained Uni-Mol Baseline

The first MSnLib Uni-Mol refine run trained with the same decoupled-KL family that we used for MassSpecGym. Exported results were:

```text
validation cosine_mean=0.56830 sqrt_cosine_mean=0.55067
test       cosine_mean=0.56170 sqrt_cosine_mean=0.54381
```

This was below FIORA and had a large low-score tail:

```text
validation sqrt fraction < 0.1 = 0.1126
test       sqrt fraction < 0.1 = 0.1210
```

Interpretation at the time would have been that the model/support was weak. The later robust fine-tune showed that was the wrong conclusion.

## MiraFrag Robust Square-Root-Cosine Fine-Tune

Starting from the KL-trained MSnLib Uni-Mol checkpoint, direct `LOSS=sqrt_cosine` fine-tuning with low learning rates produced the decisive improvement:

```text
validation n=13746
cosine_mean=0.75055
sqrt_cosine_mean=0.74146
candidate_coverage_mean=0.88720
oracle_binned_cosine_mean=0.95840
scorer_gap_mean=0.20784

test n=13683
cosine_mean=0.75813
sqrt_cosine_mean=0.74251
candidate_coverage_mean=0.88486
oracle_binned_cosine_mean=0.95934
scorer_gap_mean=0.20121
```

The low-score tail mostly disappeared:

```text
validation sqrt fraction < 0.1 = 0.02081
test       sqrt fraction < 0.1 = 0.02068
```

This beats the local FIORA precursor-excluded test result by `+0.16835` square-root cosine and even beats FIORA's default precursor-including test result by `+0.12562`.

## Conclusion

The MSnLib result changes how we should read earlier validation experiments. The largest issue was not candidate support or encoder architecture; it was objective alignment. Decoupled KL/OOS can learn useful fragment scoring but still waste probability mass or optimize a distributional target that does not match the emitted-spectrum cosine metric. Direct metric fine-tuning fixes that on MSnLib and gives a smaller but real gain on MassSpecGym.

For future MSnLib work, treat `resources/msnlib/checkpoints/mirafrag_msnlib_unimol_robust.pt` as the main checkpoint, not the earlier KL-only refine checkpoint.
