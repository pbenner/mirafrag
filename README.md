# MiraFrag

MiraFrag is a foundation-encoder package for molecular MS/MS spectrum simulation.
It can use MACE or AIMNet as the molecular encoder and replaces the energy
readout with a trainable spectrum head.

The first implementation targets the MassSpecGym spectrum simulation task:

- input: molecule structure plus precursor metadata
- model: molecular foundation encoder with a sparse fragment spectrum head
- target: observed MS/MS peaks, compared against generated fragment bins
- metrics: sparse binned cosine and square-root cosine

MiraFrag uses the MassSpecGym spectrum-simulation bin definition by default:
`0.01 Da` bins over `m/z < 1005`, i.e. `100500` output bins.

The spectrum head scores generated fragment candidates. Candidates that share an
m/z bin are aggregated before computing losses and metrics.

Precursor m/z is scaled by the configured m/z range. Collision energy is
standardized from the training split with robust median/IQR statistics, with
instrument-specific collision-energy scaling used when enough train examples are
available for an instrument.

Training uses AdamW with separate parameter groups for the MiraFrag spectrum head
and trainable encoder parameters. Both groups use `LR`; weight decay is applied
only to trainable encoder parameters. Set `SCHEDULER=none` to keep a fixed
learning rate.

Additional documentation is collected in [docs/README.md](docs/README.md).
The loss derivations for `LOSS=soft_projected_kl`, `LOSS=soft_binned_kl`,
`LOSS=soft_binned_coverage_kl`, and `LOSS=fragnnet_ce` are documented in
[docs/losses/README.md](docs/losses/README.md).

## Code Structure

The package separates the main concerns into focused modules:

- `config.py`: serializable MiraFrag model configuration
- `encoders/`: MACE and AIMNet foundation encoder adapters
- `heads/fragment.py`: candidate-based fragment spectrum head
- `checkpoint.py`: strict state-dict checkpoint save/load
- `losses.py`: sparse spectrum losses and binned/tolerance metrics
- `training.py`, `optim.py`, `evaluation.py`: training loop, optimizer/scheduler helpers, and evaluation/export
- `adducts.py`: shared adduct charge and mass parsing used by data and fragmentation

## Model

MiraFrag uses a molecular foundation model as the atom encoder and replaces the
energy readout with a sparse spectrum head. MACE is the default encoder; AIMNet
can be selected with `ENCODER=aimnet` or `--encoder aimnet`. SMILES are converted
into 3D molecular graphs with RDKit conformers, then the selected encoder
produces per-atom node features. The spectrum head pools those atom features over
each generated fragment formula and over the whole precursor molecule. Fragment
features and precursor/molecule context are encoded separately, then scored with
explicit fragment-context and fragment-collision-energy interaction terms. The
normalized collision energy also gates fragment-graph message passing.

The 3D graph construction is deterministic for a given seed. MiraFrag first asks
RDKit to turn the SMILES string into a molecule and to make all hydrogens
explicit. It then asks RDKit to propose a plausible 3D arrangement of the atoms.
Technically, this uses RDKit's ETKDGv3 conformer generator, which is a rule-based
method for placing atoms in 3D using typical bond lengths, bond angles, ring
shapes, and stereochemistry. MiraFrag tries a few increasingly permissive RDKit
settings when the default attempt fails.

After a 3D arrangement is found, MiraFrag relaxes the coordinates with a simple
classical force field. UFF is tried first because it covers many element types;
MMFF is used as a fallback when available. These force fields are not the model
being trained. They are only a preprocessing step that moves atoms into a more
reasonable geometry before the molecular encoder sees them.

MiraFrag then checks whether bonded atoms are at chemically plausible distances
using covalent radii. If relaxation creates an implausible geometry, the original
unrelaxed RDKit geometry is tried instead. If all 3D attempts fail, MiraFrag can
fall back to RDKit 2D coordinates, but that should be rare and is mainly a
robustness path. Encoder graph edges connect atom pairs that are closer than the
selected encoder cutoff radius.

The metadata vector contains scaled precursor m/z, standardized collision
energy, an adduct embedding, and an instrument-type embedding. Collision energy
is standardized from the training split with robust median/IQR statistics, with
instrument-specific scaling when enough examples are available.

The fragment head is candidate based:

- fragment atom features are mean-pooled from encoder node features
- whole-molecule atom features are mean-pooled as precursor context
- fragment features and precursor metadata/molecule context are encoded separately
- optional message passing runs over a fragment graph
- normalized collision energy conditions fragment-graph messages through feature-wise modulation and edge gates
- each fragment formula is scored from fragment features, precursor context, collision-energy features, and multiplicative interaction terms
- isotope/adduct peak priors are added as log priors
- an out-of-support (OOS) logit models target peaks with no generated candidate

Predictions are sparse. Multiple candidates that land in the same m/z bin are
aggregated before computing binned losses, metrics, and exported spectra.

Encoder adaptation is controlled by `FINE_TUNE_STRATEGY`:

- `head`: freeze the encoder and train only the spectrum head
- `delta`: freeze encoder base weights and train additive delta weights
- `full`: train all encoder weights and the spectrum head

The optimizer uses AdamW with separate parameter groups. The spectrum head uses
no weight decay; weight decay applies only to trainable encoder or delta
parameters.

## Fragmentation

Fragment candidates are generated from the RDKit molecule before each model
forward pass, or loaded from the feature cache. The generator is adapted from
FraGNNet's fragment-tree idea, but MiraFrag keeps its own implementation and uses
the generated candidates as sparse support for the spectrum head.

The fragmentation procedure is:

1. Parse the SMILES with RDKit and compute implicit/explicit hydrogen counts.
2. Build a recursive fragment tree by removing one atom at a time.
3. After removing an atom, keep connected components as child fragments.
4. Collapse isomorphic fragments using a Weisfeiler-Lehman-style hash.
5. Track tree depth and cumulative broken-bond budget.
6. Enumerate hydrogen transfers within the allowed budget.
7. Convert each fragment formula to neutral mass, then add the adduct mass.
8. Optionally expand formulas into isotope peaks and add isotope log priors.
9. Prune to `MAX_FRAGMENTS` formulas and `MAX_FRAGMENT_EDGES` graph edges.

The default fragment controls are exposed in the Makefile. The tree depth and
broken-bond budget match FraGNNet's D3-style preprocessing defaults: depth 3 and
broken-bond budget 6. For D4-style runs, set `MAX_FRAGMENT_TREE_DEPTH=4`. MiraFrag
keeps at most 1 isotope peak per formula by default.

- `MAX_FRAGMENT_TREE_DEPTH`: recursive atom-removal depth, default `3`
- `MAX_FRAGMENT_BROKEN_BONDS`: cumulative broken-bond and hydrogen-transfer budget, default `6`
- `MAX_FRAGMENTS`: maximum retained fragment formulas per molecule, default `2048`
- `MAX_FRAGMENT_EDGES`: maximum fragment-graph message-passing edges, default `8192`
- `INCLUDE_FRAGMENT_ISOTOPES`: include isotope peak candidates, default `1`
- `MAX_FRAGMENT_ISOTOPE_PEAKS`: maximum isotope peaks per formula, default `1`

The fragment graph connects related formulas with directed typed edges:
parent/child tree edges, same-formula isotope neighbors, same atom-set hydrogen
transfer neighbors, and local same-bin neighbors. Edge features encode the
relation type plus normalized mass, hydrogen-shift, and isotope-rank deltas.

Fragment adduct parsing supports bracketed signed adducts such as `[M+H]+`,
`[M+Na]+`, `[M-H]-`, `[M+2H]2+`, and formula modifiers such as `[M+FA-H]-`.
Unsupported adduct syntax raises an error rather than silently using the wrong
mass.

## Install

Install the package environment from the repository root:

```bash
uv sync --extra mace
```

For AIMNet instead, use:

```bash
uv sync --extra aimnet
```

The MassSpecGym workflow lives in `resources/massspecgym`. Run its Makefile from
that folder, or with `make -C resources/massspecgym ...`. The workflow keeps its
data, checkpoints, predictions, and feature cache under `resources/massspecgym/`.

Use `make -C resources/massspecgym prepare-data` to download the MassSpecGym TSV
into `resources/massspecgym/data/`. Use `make -C resources/massspecgym
prepare-cache` to precompute graph and fragment caches before training or
evaluation; the MassSpecGym `train`, `eval`, and `predict` targets pass the
same disk cache directory by default. Disk caching is controlled by
`DISK_CACHE_DIR` / `--disk-cache-dir`; per-worker in-memory reuse is controlled
by `MEMORY_CACHE` / `--memory-cache`. When a training disk cache directory is
configured, the `train` target pre-fills missing train/val cache entries with
explicit `cache train` and `cache val` tqdm bars before the first epoch. Cache
filling uses an unordered multiprocessing pool, so each worker receives another
sample as soon as it finishes instead of waiting for earlier slow samples.
`prepare-cache` uses the foundation model and the Makefile fragment settings
unless you pass `CACHE_SOURCE_MODEL=resources/massspecgym/checkpoints/mirafrag.pt`.

## Train

The recommended training entry point is the Makefile. It uses the MassSpecGym
simulation-challenge filter, `LOSS=kl`, the configured feature cache, and full
encoder fine-tuning by default:

```bash
make -C resources/massspecgym train LOSS=kl LR=1e-5
```

To train with AIMNet instead of MACE:

```bash
make -C resources/massspecgym train ENCODER=aimnet LOSS=kl LR=1e-5
```

Direct CLI use is also supported. The direct CLI default is conservative
head-only training, so pass `--fine-tune-strategy full` when you want full
encoder fine-tuning:

```bash
mirafrag-train \
  --input resources/massspecgym/data/MassSpecGym.tsv \
  --output resources/massspecgym/checkpoints/mirafrag.pt \
  --encoder mace \
  --foundation-source off \
  --foundation-model medium \
  --fine-tune-strategy full \
  --loss kl \
  --epochs 20 \
  --batch-size 8
```

If `--input` is omitted and `massspecgym` is installed, MiraFrag asks MassSpecGym
to load the benchmark data.

## Evaluate

```bash
mirafrag-eval \
  --input /path/to/MassSpecGym.tsv \
  --model resources/massspecgym/checkpoints/mirafrag.pt \
  --split test \
  --output resources/massspecgym/predictions/massspecgym_test_predictions.csv
```

## Predict

```bash
mirafrag-predict \
  --input molecules.csv \
  --model resources/massspecgym/checkpoints/mirafrag.pt \
  --output resources/massspecgym/predictions.csv
```

The input table should contain at least `smiles`. For best results include
MassSpecGym-style metadata columns: `precursor_mz`, `adduct`,
`instrument_type`, and `collision_energy`.
Fragment candidates are generated with adduct-specific ion masses for bracketed
signed adducts such as `[M+H]+`, `[M+Na]+`, `[M-H]-`, and `[M+2H]2+`.
