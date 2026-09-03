from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import shutil
import signal
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from mirafrag.cache_fill import prefill_feature_cache
from mirafrag.checkpoint import load_checkpoint, save_checkpoint
from mirafrag.chem import infer_graph_config, quiet_rdkit_logs
from mirafrag.cli.common import (
    add_high_ce_fragment_support_args,
    apply_fragment_args_to_model_config,
    resolve_device,
    validate_checkpoint_bin_config,
)
from mirafrag.config import MiraFragConfig
from mirafrag.data import (
    ADDUCT_ALIASES,
    CE_ALIASES,
    INSTRUMENT_ALIASES,
    PRECURSOR_ALIASES,
    RAW_COLLISION_ENERGY_COLUMN,
    SAMPLE_WEIGHT_COLUMN,
    SMILES_ALIASES,
    BinnedSpectrumDataset,
    MetadataConfig,
    collate_spectrum_batch,
    dataloader_performance_kwargs,
    filter_massspecgym_simulation,
    filter_supported_elements,
    find_column,
    merge_group_spectra,
    normalize_collision_energy_dataframe,
    parse_physical_bond_feature_columns,
    read_table,
    select_split,
)
from mirafrag.encoders import load_foundation_encoder
from mirafrag.fragments import fragment_support_profile_from_model_config
from mirafrag.losses import LOSS_NAMES
from mirafrag.model import MiraFragModel, set_encoder_finetune_strategy
from mirafrag.retrieval import (
    build_retrieval_candidate_rows,
    read_retrieval_candidate_table,
)
from mirafrag.sparse_spectra import (
    SparseSpectrum,
    combine_sparse_spectra,
    scale_sparse_spectrum,
    sparse_cosine,
    sparse_from_peaks,
)
from mirafrag.spectra import (
    MASS_SPEC_GYM_BIN_WIDTH,
    MASS_SPEC_GYM_MZ_MAX,
    num_spectrum_bins,
    parse_peaks,
)
from mirafrag.training import run_epoch, train_model


@dataclass(frozen=True)
class _ValidationTuneCandidate:
    """
    One validation-tuning trial configuration.
    """

    head_lr: float
    encoder_lr: float
    dropout: float
    head_weight_decay: float
    encoder_weight_decay: float
    swa: bool
    swa_start_epoch: int | None
    swa_lr: float | None
    swa_anneal_epochs: int


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for MiraFrag training.

    The parser covers encoder selection, checkpoint resume, fragment-support settings, losses, optimizer and scheduler controls, cache behavior, and MassSpecGym split selection.
    """
    parser = argparse.ArgumentParser(
        prog='mirafrag-train',
        description='Fine-tune a foundation encoder with a spectrum prediction head.',
    )
    parser.add_argument('-i', '--input', default=None, help='MassSpecGym TSV/CSV path.')
    parser.add_argument('-o', '--output', required=True, help='Output checkpoint .pt.')
    parser.add_argument(
        '--init-checkpoint',
        default=None,
        help=(
            'Optional MiraFrag checkpoint to continue from. The saved spectrum head, '
            'metadata, and foundation weights are reused; --fine-tune-strategy '
            'controls how the encoder is adapted for the new run.'
        ),
    )
    parser.add_argument('--foundation-source', default='off')
    parser.add_argument('--foundation-model', default='medium')
    parser.add_argument('--foundation-path', default=None)
    parser.add_argument(
        '--encoder',
        choices=['mace', 'aimnet', 'unimol', 'small3d'],
        default=None,
        help='Foundation atom encoder.',
    )
    parser.add_argument(
        '--aimnet-model',
        default='aimnet2',
        help='AIMNet registry model name used when --encoder aimnet.',
    )
    parser.add_argument(
        '--aimnet-path',
        default=None,
        help='Optional local AIMNet model path used when --encoder aimnet.',
    )
    parser.add_argument('--unimol-model-name', default='unimolv1')
    parser.add_argument('--unimol-model-size', default='84m')
    parser.add_argument('--unimol-pretrained-model-path', default=None)
    parser.add_argument('--unimol-pretrained-dict-path', default=None)
    parser.add_argument('--unimol-max-atoms', type=int, default=512)
    parser.add_argument(
        '--unimol-mode',
        choices=['trainable', 'frozen'],
        default='trainable',
        help='Uni-Mol encoder mode: trainable uses the underlying nn.Module; frozen uses UniMolRepr inference.',
    )
    parser.add_argument(
        '--graph-relaxation',
        choices=['rdkit', 'aimnet', 'none'],
        default='rdkit',
        help='Coordinate relaxation method used when building molecular graphs.',
    )
    parser.add_argument(
        '--aimnet-relax-model',
        default=None,
        help='AIMNet model name used for --graph-relaxation aimnet.',
    )
    parser.add_argument(
        '--aimnet-relax-steps',
        type=int,
        default=50,
        help='Maximum ASE optimizer steps for AIMNet graph relaxation.',
    )
    parser.add_argument(
        '--aimnet-relax-fmax',
        type=float,
        default=0.05,
        help='ASE force convergence threshold for AIMNet graph relaxation.',
    )
    parser.add_argument(
        '--aimnet-relax-device',
        default='auto',
        help='Device for AIMNet graph relaxation; auto lets AIMNet choose.',
    )
    parser.add_argument('--device', default='auto')
    parser.add_argument(
        '--collision-energy-mode',
        choices=['raw', 'normalized'],
        default='raw',
        help=(
            'Use raw CE with model-side robust normalization, or pre-normalize CE '
            'from training statistics and pass it directly to the model.'
        ),
    )
    parser.add_argument(
        '--merge-group-spectra',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Merge replicate spectra inside each selected split by molecule, '
            'adduct, and instrument before training/evaluation.'
        ),
    )
    parser.add_argument(
        '--merge-group-cols',
        default='auto',
        help=(
            'Comma-separated dataframe columns used by --merge-group-spectra; '
            'auto uses SMILES, adduct, and instrument when available.'
        ),
    )
    parser.add_argument(
        '--group-balanced-loss',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Weight training examples by inverse replicate group frequency. '
            'Validation remains unweighted.'
        ),
    )
    parser.add_argument(
        '--group-balance-cols',
        default='auto',
        help=(
            'Comma-separated dataframe columns used by --group-balanced-loss; '
            'auto uses SMILES, adduct, and instrument when available.'
        ),
    )
    parser.add_argument(
        '--group-balance-power',
        type=float,
        default=1.0,
        help='Exponent for inverse group-size weighting; 1.0 gives equal total weight per group.',
    )
    parser.add_argument(
        '--metadata-ce-interaction',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            'Add a zero-initialized CE x instrument-type residual to metadata '
            'conditioning. This preserves checkpoint predictions initially.'
        ),
    )
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument(
        '--head-learning-rate',
        type=float,
        default=None,
        help='Optional spectrum-head learning rate; defaults to --learning-rate.',
    )
    parser.add_argument(
        '--encoder-learning-rate',
        type=float,
        default=None,
        help='Optional trainable-encoder learning rate; defaults to --learning-rate.',
    )
    parser.add_argument(
        '--encoder-layer-lr-decay',
        type=float,
        default=1.0,
        help=(
            'Optional AIMNet encoder layerwise learning-rate decay in (0, 1]. '
            'The final AIMNet MLP block uses the encoder learning rate; earlier '
            'blocks and base encoder parameters receive progressively smaller rates.'
        ),
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-5,
        help=(
            'Legacy trainable-encoder weight decay. Use --encoder-weight-decay '
            'to set it explicitly; the spectrum head is controlled separately '
            'with --head-weight-decay.'
        ),
    )
    parser.add_argument(
        '--head-weight-decay',
        type=float,
        default=0.0,
        help='AdamW weight decay for decayable spectrum-head weight matrices.',
    )
    parser.add_argument(
        '--head-delta-regularization',
        type=float,
        default=0.0,
        help=(
            'Checkpoint-centered L2 penalty on trainable spectrum-head weight '
            'matrices. This keeps fine-tuning close to the initial checkpoint; '
            '0 disables it.'
        ),
    )
    parser.add_argument(
        '--encoder-weight-decay',
        type=float,
        default=None,
        help='AdamW weight decay for decayable trainable-encoder weight matrices.',
    )
    parser.add_argument(
        '--encoder-delta-regularization',
        type=float,
        default=0.0,
        help=(
            'Checkpoint-centered L2 penalty on trainable encoder weight matrices. '
            'This keeps encoder fine-tuning close to the initial checkpoint; '
            '0 disables it.'
        ),
    )
    parser.add_argument(
        '--scheduler',
        choices=['exponential', 'plateau', 'constant', 'cosine', 'none'],
        default='exponential',
        help='Learning-rate scheduler.',
    )
    parser.add_argument(
        '--scheduler-interval',
        choices=['epoch', 'step'],
        default='epoch',
        help='Run step/epoch schedulers once per epoch or once per batch.',
    )
    parser.add_argument(
        '--min-lr-ratio',
        type=float,
        default=0.1,
        help="Final LR as a fraction of each parameter group's initial LR.",
    )
    parser.add_argument(
        '--exponential-gamma',
        type=float,
        default=0.8,
        help='Multiplicative LR decay per scheduler update for --scheduler exponential.',
    )
    parser.add_argument(
        '--plateau-factor',
        type=float,
        default=0.5,
        help='LR multiplier used by --scheduler plateau.',
    )
    parser.add_argument(
        '--plateau-patience',
        type=int,
        default=2,
        help='Validation epochs without improvement before plateau LR decay.',
    )
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--metadata-dim', type=int, default=32)
    parser.add_argument(
        '--max-fragment-tree-depth',
        type=int,
        default=None,
        help='Maximum recursive atom-removal tree depth.',
    )
    parser.add_argument(
        '--max-fragment-broken-bonds',
        type=int,
        default=None,
        help=(
            'Maximum cumulative broken-bond order and hydrogen-transfer budget '
            'for recursive fragmentation.'
        ),
    )
    parser.add_argument(
        '--max-fragments',
        type=int,
        default=None,
        help='Maximum fragment candidates per molecule.',
    )
    parser.add_argument(
        '--max-fragment-edges',
        type=int,
        default=None,
        help='Maximum directed fragment-relationship edges per molecule.',
    )
    parser.add_argument(
        '--fragment-gnn-layers',
        type=int,
        default=2,
        help='Number of message-passing layers over the fragment graph.',
    )
    parser.add_argument(
        '--fragment-path-layers',
        type=int,
        default=None,
        help=(
            'Number of recursive parent-to-child path propagation steps in the '
            'fragment head; 0 disables the path scorer.'
        ),
    )
    parser.add_argument(
        '--fragment-bond-break-layers',
        type=int,
        default=None,
        help=(
            'Number of hidden layers in the optional FIORA-style bond-event auxiliary scorer; '
            '0 disables explicit cut-bond scoring. When enabled, fragment caches '
            'include candidate cut-bond provenance.'
        ),
    )
    parser.add_argument(
        '--fragment-path-primary',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            'Use learned root-to-fragment path probabilities as the primary '
            'formula scorer. This keeps multi-break fragments by accumulating '
            'transition scores over the retained fragment DAG.'
        ),
    )
    parser.add_argument(
        '--fragment-action-path-layers',
        type=int,
        default=None,
        help=(
            'Number of hidden layers in the constrained local action-path '
            'residual. This scores parent-to-child fragmentation actions from '
            'hand-crafted fragment/edge descriptors and adds a zero-initialized '
            'residual on top of the existing formula scorer.'
        ),
    )
    parser.add_argument(
        '--fragment-action-primary-layers',
        type=int,
        default=None,
        help=(
            'Number of hidden layers in the explicit bond-action formula residual. '
            'This uses cached oriented broken-bond provenance to inject latent '
            'multi-break fragmentation evidence into the primary formula logits.'
        ),
    )
    parser.add_argument(
        '--fragment-action-primary-ce-gate',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            'Enable a zero-initialized collision-energy gate on aggregated '
            'action-primary evidence. This lets the explicit bond-action branch '
            'learn CE-dependent fragmentation strength while preserving loaded '
            'checkpoint predictions before fine-tuning.'
        ),
    )
    parser.add_argument(
        '--bond-break-geometry-features',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            'Append invariant conformer geometry features to explicit broken-bond '
            'events. When unset, new bond-action branches enable this by default.'
        ),
    )
    parser.add_argument(
        '--fragment-action-geometry-features',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=('Deprecated alias for --bond-break-geometry-features.'),
    )
    parser.add_argument(
        '--fragment-action-bond-gnn-layers',
        type=int,
        default=None,
        help=(
            'Number of grouped message-passing layers over broken-bond events '
            'inside each candidate formula. This adds a zero-initialized '
            'multi-break local chemistry residual on top of formula logits.'
        ),
    )
    parser.add_argument(
        '--fragnnet-dag-layers',
        type=int,
        default=None,
        help=(
            'Enable a FraGNNet-style primary fragment-DAG scorer with this many '
            'message-passing layers. The branch scores retained formula nodes '
            'from connected-component atom features, formula composition, depth, '
            'hydrogen shift, metadata, and DAG edges.'
        ),
    )
    parser.add_argument(
        '--fragnnet-dag-num-hs',
        type=int,
        default=None,
        help=(
            'Maximum absolute hydrogen-shift slot predicted by the FraGNNet-style '
            'DAG scorer. The default 4 matches the FraGNNet D3/D4 configs.'
        ),
    )
    parser.add_argument(
        '--ce-basis-features',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            'Use a smooth nonlinear collision-energy basis for CE-conditioned '
            'fragment scoring instead of the scalar normalized CE channel.'
        ),
    )
    parser.add_argument(
        '--ce-embedding',
        choices=['scalar', 'basis', 'fourier'],
        default=None,
        help=(
            'Collision-energy feature transform for CE-conditioned fragment '
            'scoring. The legacy --ce-basis-features flag is equivalent to '
            '--ce-embedding basis when this option is not set.'
        ),
    )
    parser.add_argument(
        '--ce-fourier-frequencies',
        type=int,
        default=None,
        help='Number of powers-of-two Fourier frequencies for --ce-embedding fourier.',
    )
    parser.add_argument(
        '--aimnet-charge-features',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            'Add a zero-initialized fragment residual from AIMNet per-atom '
            'charge summaries. Only AIMNet encoders export these auxiliary features.'
        ),
    )
    parser.add_argument(
        '--aimnet-multipass-features',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            'Fuse intermediate AIMNet message-passing atom states through a '
            'zero-initialized node-feature residual before fragment scoring.'
        ),
    )
    parser.add_argument(
        '--encoder-bond-adapter-layers',
        type=int,
        default=None,
        help=(
            'Number of residual molecular-edge message-passing layers applied '
            'to encoder node features before fragment scoring; 0 disables it.'
        ),
    )
    parser.add_argument(
        '--encoder-bond-adapter-feature-dim',
        type=int,
        default=None,
        help='Hidden dimension for --encoder-bond-adapter-layers.',
    )
    parser.add_argument(
        '--encoder-bond-adapter-dropout',
        type=float,
        default=None,
        help='Dropout used inside --encoder-bond-adapter-layers.',
    )
    parser.add_argument(
        '--encoder-metadata-adapter-layers',
        type=int,
        default=None,
        help=(
            'Number of residual CE/adduct/instrument-conditioned message-passing '
            'layers applied to encoder node features; 0 disables it.'
        ),
    )
    parser.add_argument(
        '--encoder-metadata-adapter-feature-dim',
        type=int,
        default=None,
        help='Hidden dimension for --encoder-metadata-adapter-layers.',
    )
    parser.add_argument(
        '--encoder-metadata-adapter-dropout',
        type=float,
        default=None,
        help='Dropout used inside --encoder-metadata-adapter-layers.',
    )
    parser.add_argument(
        '--candidate-transformer-layers',
        type=int,
        default=None,
        help=(
            'Number of residual self-attention layers over top-scoring fragment '
            'candidates per spectrum; 0 disables candidate-set interactions.'
        ),
    )
    parser.add_argument(
        '--candidate-transformer-heads',
        type=int,
        default=None,
        help='Attention heads for --candidate-transformer-layers.',
    )
    parser.add_argument(
        '--candidate-transformer-max-tokens',
        type=int,
        default=None,
        help=(
            'Maximum top-scoring formula candidates per spectrum attended by '
            '--candidate-transformer-layers.'
        ),
    )
    parser.add_argument(
        '--candidate-suppression-gate-hidden-dim',
        type=int,
        default=None,
        help=(
            'Hidden dimension for a suppression-only candidate gate applied to '
            'formula logits before peak expansion; 0 disables it.'
        ),
    )
    parser.add_argument(
        '--candidate-suppression-gate-dropout',
        type=float,
        default=None,
        help='Dropout used inside --candidate-suppression-gate-hidden-dim.',
    )
    parser.add_argument(
        '--candidate-suppression-gate-initial-penalty',
        type=float,
        default=None,
        help=(
            'Tiny initial positive penalty subtracted back out so the gate starts '
            'as a no-op while remaining suppression-only after training.'
        ),
    )
    parser.add_argument(
        '--conditional-expert-heads',
        type=int,
        default=None,
        help=(
            'Number of internally gated residual formula-scoring experts; 1 disables '
            'the conditional expert branch.'
        ),
    )
    parser.add_argument(
        '--conditional-expert-hidden-dim',
        type=int,
        default=None,
        help='Hidden dimension for --conditional-expert-heads.',
    )
    parser.add_argument(
        '--conditional-expert-dropout',
        type=float,
        default=None,
        help='Dropout used inside --conditional-expert-heads.',
    )
    parser.add_argument(
        '--molecule-descriptor-features',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            'Add zero-initialized formula/OOS residuals conditioned on cheap RDKit '
            'whole-molecule descriptors. This targets support/OOS regimes without '
            'changing graph or fragment cache keys.'
        ),
    )
    parser.add_argument(
        '--molecule-descriptor-hidden-dim',
        type=int,
        default=None,
        help='Hidden dimension for --molecule-descriptor-features.',
    )
    parser.add_argument(
        '--molecule-descriptor-dropout',
        type=float,
        default=None,
        help='Dropout used inside --molecule-descriptor-features.',
    )
    parser.add_argument(
        '--include-fragment-isotopes',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Expand fragment formula candidates into isotope peak candidates.',
    )
    parser.add_argument(
        '--fragment-isotope-threshold',
        type=float,
        default=None,
        help='Minimum retained isotope probability per fragment formula.',
    )
    parser.add_argument(
        '--max-fragment-isotope-peaks',
        type=int,
        default=None,
        help='Maximum isotope peaks retained per fragment formula.',
    )
    parser.add_argument(
        '--physical-bond-features',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Append cid-physical-features bond sidecar descriptors to fragment broken-bond features.',
    )
    parser.add_argument(
        '--physical-bond-features-path',
        default=None,
        help='CSV sidecar from cid-physical-features; required when --physical-bond-features is enabled.',
    )
    parser.add_argument(
        '--physical-bond-feature-columns',
        default=None,
        help='Comma-separated sidecar columns to append; defaults to the recommended RDKit/geometry set.',
    )
    add_high_ce_fragment_support_args(parser)
    parser.add_argument(
        '--fine-tune-strategy',
        choices=['head', 'delta', 'full', 'aimnet-final'],
        default='head',
        help=(
            'Encoder adaptation strategy: head freezes the encoder, delta trains '
            'additive residual parameters, full trains encoder weights, and '
            'aimnet-final trains only AIMNet2 model.mlps.2.6.'
        ),
    )
    parser.add_argument('--mz-max', type=float, default=MASS_SPEC_GYM_MZ_MAX)
    parser.add_argument('--bin-width', type=float, default=MASS_SPEC_GYM_BIN_WIDTH)
    parser.add_argument(
        '--loss',
        choices=LOSS_NAMES,
        default='decoupled_kl',
    )
    parser.add_argument(
        '--kl-weight',
        type=float,
        default=0.7,
        help='KL mixture weight for --loss kl_cosine or decoupled_kl_cosine; 1.0 is pure KL.',
    )
    parser.add_argument(
        '--coverage-weight',
        type=float,
        default=0.1,
        help='Coverage penalty weight for --loss soft_binned_coverage_kl.',
    )
    parser.add_argument(
        '--target-power',
        type=float,
        default=1.0,
        help=(
            'Sharpen target intensities before KL normalization. '
            '1.0 keeps standard intensity-normalized KL.'
        ),
    )
    parser.add_argument(
        '--entropy-weight',
        type=float,
        default=0.0,
        help='Optional entropy penalty on predicted fragment-plus-OOS probabilities.',
    )
    parser.add_argument(
        '--target-neighbor-smoothing',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Train against row targets mixed with same-molecule nearby-CE '
            'empirical replicate spectra. Validation targets are never smoothed.'
        ),
    )
    parser.add_argument(
        '--target-neighbor-weight',
        type=float,
        default=0.3,
        help='Mixture weight for neighbor empirical targets when smoothing is enabled.',
    )
    parser.add_argument(
        '--target-neighbor-ce-window',
        type=float,
        default=10.0,
        help='Maximum raw collision-energy distance for target-neighbor smoothing.',
    )
    parser.add_argument(
        '--target-neighbor-same-instrument',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Restrict target-neighbor smoothing to the same instrument.',
    )
    parser.add_argument(
        '--target-neighbor-same-adduct',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Restrict target-neighbor smoothing to the same adduct.',
    )
    parser.add_argument(
        '--retrieval-calibration-head',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Enable the optional zero-initialized retrieval calibration logit head.',
    )
    parser.add_argument(
        '--distill-mode',
        choices=['fixed', 'rowwise-oracle'],
        default='fixed',
        help=(
            'Teacher spectrum construction mode. fixed blends all teacher files '
            'with --distill-weights; rowwise-oracle requires two teacher files and '
            'weights them per training row by their precursor-excluded target cosine.'
        ),
    )
    parser.add_argument(
        '--distill-predictions',
        default=None,
        help=(
            'Comma-separated teacher prediction CSVs with identifier and pred_peaks; '
            'multiple files are blended using --distill-weights.'
        ),
    )
    parser.add_argument(
        '--distill-weights',
        default=None,
        help=(
            'Colon- or comma-separated teacher weights matching --distill-predictions; '
            'defaults to equal weights.'
        ),
    )
    parser.add_argument(
        '--distill-oracle-temperature',
        type=float,
        default=0.03,
        help=(
            'Softmax temperature for --distill-mode rowwise-oracle. Smaller values '
            'approach hard per-row teacher selection; 0 uses hard selection.'
        ),
    )
    parser.add_argument(
        '--distill-oracle-min-teacher-weight',
        type=float,
        default=0.05,
        help=(
            'Minimum per-teacher mixture weight for --distill-mode rowwise-oracle; '
            'set 0 for an unconstrained soft oracle.'
        ),
    )
    parser.add_argument(
        '--distill-loss-weight',
        type=float,
        default=0.0,
        help='Weight for projected teacher-spectrum KL distillation.',
    )
    parser.add_argument(
        '--distill-min-overlap-mass',
        type=float,
        default=0.05,
        help=(
            'Minimum teacher probability mass that must overlap generated fragment '
            'bins before applying the distillation term for a sample.'
        ),
    )
    parser.add_argument(
        '--distill-ce-min',
        type=float,
        default=None,
        help='Only apply distillation to training rows with raw collision energy >= this value.',
    )
    parser.add_argument(
        '--distill-ce-max',
        type=float,
        default=None,
        help='Only apply distillation to training rows with raw collision energy <= this value.',
    )
    parser.add_argument(
        '--retrieval-candidate-input',
        default=None,
        help='Optional explicit retrieval candidate table/JSON for auxiliary ranking training.',
    )
    parser.add_argument(
        '--retrieval-loss-weight',
        type=float,
        default=0.0,
        help='Weight for auxiliary groupwise retrieval cross-entropy training.',
    )
    parser.add_argument(
        '--retrieval-negatives',
        type=int,
        default=16,
        help='Number of decoy candidates per query for auxiliary retrieval training.',
    )
    parser.add_argument(
        '--retrieval-max-queries',
        type=int,
        default=None,
        help='Optional cap on training queries used for auxiliary retrieval training.',
    )
    parser.add_argument(
        '--retrieval-num-workers',
        type=int,
        default=None,
        help='Worker count for auxiliary retrieval batches; defaults to --num-workers.',
    )
    parser.add_argument(
        '--retrieval-prefill-cache',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Precompute auxiliary retrieval candidate feature cache before training.',
    )
    parser.add_argument(
        '--retrieval-cache-num-workers',
        type=int,
        default=None,
        help='Worker count for auxiliary retrieval cache prefill; defaults to --retrieval-num-workers.',
    )
    parser.add_argument(
        '--retrieval-cache-chunk-size',
        type=int,
        default=1,
        help='Chunk size for auxiliary retrieval cache prefill.',
    )
    parser.add_argument(
        '--retrieval-sample-timeout',
        type=float,
        default=120.0,
        help='Per-candidate feature timeout in seconds for auxiliary retrieval training; 0 disables.',
    )
    parser.add_argument('--train-split', default='train')
    parser.add_argument('--val-split', default='val')
    parser.add_argument('--split-col', default='auto')
    parser.add_argument(
        '--mass-tolerance',
        type=float,
        default=0.01,
        help=(
            'Absolute Da tolerance for tolerance-based losses. '
            'For --loss soft_projected_kl, sigma is half this value.'
        ),
    )
    parser.add_argument(
        '--relative-mass-tolerance',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Interpret --mass-tolerance as a relative tolerance.',
    )
    parser.add_argument(
        '--mass-tolerance-min-mz',
        type=float,
        default=200.0,
        help='Minimum m/z divisor for relative tolerance matching.',
    )
    parser.add_argument('--train-split-value', default=None)
    parser.add_argument('--val-split-value', default=None)
    parser.add_argument(
        '--massspecgym-filter',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Apply the MassSpecGym simulation-challenge filter.',
    )
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument(
        '--memory-cache', action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        '--disk-cache-dir',
        default=None,
        help='Optional disk cache for precomputed encoder graphs and fragment candidates.',
    )
    parser.add_argument(
        '--prefill-cache',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Precompute missing disk-cache entries before constructing DataLoaders.',
    )
    parser.add_argument(
        '--cache-num-workers',
        type=int,
        default=None,
        help='Worker count for disk-cache prefill. Defaults to --num-workers.',
    )
    parser.add_argument(
        '--cache-chunk-size',
        type=int,
        default=None,
        help='Chunk size for disk-cache prefill. Defaults to 1.',
    )
    parser.add_argument(
        '--slow-sample-seconds',
        type=float,
        default=0.0,
        help=(
            'Print idx/identifier/SMILES diagnostics for dataset samples taking '
            'at least this many seconds. Use 0 to disable.'
        ),
    )
    parser.add_argument(
        '--trace-samples',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Print every dataset sample as it starts loading.',
    )
    parser.add_argument(
        '--dataloader-timeout',
        type=float,
        default=0.0,
        help='Seconds before a DataLoader worker timeout; 0 disables the timeout.',
    )
    parser.add_argument(
        '--progress',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Show tqdm progress bars during training and validation.',
    )
    parser.add_argument(
        '--checkpoint-metric',
        choices=['val_loss', 'train_loss', 'val_cosine', 'train_cosine'],
        default='val_loss',
        help='Metric used to decide when to save the best checkpoint.',
    )
    parser.add_argument(
        '--swa',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Enable stochastic weight averaging after --swa-start-epoch.',
    )
    parser.add_argument(
        '--swa-start-epoch',
        type=int,
        default=None,
        help='First epoch included in the SWA average; defaults to epochs // 2.',
    )
    parser.add_argument(
        '--swa-lr',
        type=float,
        default=None,
        help='Optional SWA tail learning rate. If omitted, the main scheduler continues.',
    )
    parser.add_argument(
        '--swa-anneal-epochs',
        type=int,
        default=1,
        help='Epochs used by PyTorch SWALR to anneal to --swa-lr.',
    )
    parser.add_argument(
        '--validation-tune',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Run multiple short fine-tuning trials from --init-checkpoint and '
            'copy the best validation checkpoint to --output.'
        ),
    )
    parser.add_argument(
        '--tune-metric',
        choices=['val_cosine', 'val_loss'],
        default='val_cosine',
        help='Validation metric optimized by --validation-tune.',
    )
    parser.add_argument(
        '--tune-trials',
        type=int,
        default=12,
        help='Maximum number of validation-tuning trials.',
    )
    parser.add_argument(
        '--tune-epochs',
        type=int,
        default=12,
        help='Epochs per validation-tuning trial.',
    )
    parser.add_argument(
        '--tune-lrs',
        default='1e-5,3e-5,1e-4',
        help=(
            'Comma-separated learning rates sampled by --validation-tune. Used '
            'for head and encoder unless their specific tune lists are set.'
        ),
    )
    parser.add_argument(
        '--tune-head-lrs',
        default=None,
        help='Comma-separated head learning rates sampled by --validation-tune.',
    )
    parser.add_argument(
        '--tune-encoder-lrs',
        default=None,
        help='Comma-separated encoder learning rates sampled by --validation-tune.',
    )
    parser.add_argument(
        '--tune-dropouts',
        default='0,0.02,0.05',
        help='Comma-separated head dropout probabilities sampled by --validation-tune.',
    )
    parser.add_argument(
        '--tune-weight-decays',
        default='0,1e-6,1e-5',
        help=(
            'Comma-separated encoder weight decay values sampled by '
            '--validation-tune unless --tune-encoder-weight-decays is set.'
        ),
    )
    parser.add_argument(
        '--tune-head-weight-decays',
        default=None,
        help='Comma-separated head weight decays sampled by --validation-tune.',
    )
    parser.add_argument(
        '--tune-encoder-weight-decays',
        default=None,
        help='Comma-separated encoder weight decays sampled by --validation-tune.',
    )
    parser.add_argument(
        '--tune-swa-start-epochs',
        default='3,6',
        help='Comma-separated SWA start epochs sampled by --validation-tune.',
    )
    parser.add_argument(
        '--tune-swa-lrs',
        default='1e-5,3e-5,1e-4',
        help='Comma-separated SWA learning rates sampled by --validation-tune.',
    )
    parser.add_argument(
        '--tune-seed',
        type=int,
        default=None,
        help='Random seed for validation-tuning trial sampling; defaults to --seed.',
    )
    parser.add_argument(
        '--tune-output-dir',
        default=None,
        help='Directory for per-trial checkpoints; defaults next to --output.',
    )
    parser.add_argument(
        '--tune-keep-trials',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Keep per-trial checkpoints after copying the best one to --output.',
    )
    return parser.parse_args()


def main() -> None:
    """
    Run the end-to-end training command.

    This entry point reads and filters data, creates or restores a model, prepares datasets and caches, builds DataLoaders, and calls the training loop with the resolved configuration.
    """
    args = parse_args()
    if args.fragment_path_layers is not None and args.fragment_path_layers < 0:
        raise SystemExit('--fragment-path-layers must be nonnegative.')
    if (
        args.fragment_bond_break_layers is not None
        and args.fragment_bond_break_layers < 0
    ):
        raise SystemExit('--fragment-bond-break-layers must be nonnegative.')
    if (
        args.fragment_action_path_layers is not None
        and args.fragment_action_path_layers < 0
    ):
        raise SystemExit('--fragment-action-path-layers must be nonnegative.')
    if (
        args.fragment_action_primary_layers is not None
        and args.fragment_action_primary_layers < 0
    ):
        raise SystemExit('--fragment-action-primary-layers must be nonnegative.')
    if (
        args.fragment_action_bond_gnn_layers is not None
        and args.fragment_action_bond_gnn_layers < 0
    ):
        raise SystemExit('--fragment-action-bond-gnn-layers must be nonnegative.')
    if args.fragnnet_dag_layers is not None and args.fragnnet_dag_layers < 0:
        raise SystemExit('--fragnnet-dag-layers must be nonnegative.')
    if args.fragnnet_dag_num_hs is not None and args.fragnnet_dag_num_hs < 0:
        raise SystemExit('--fragnnet-dag-num-hs must be nonnegative.')
    if (
        args.fragment_path_primary is True
        and not args.init_checkpoint
        and not args.fragment_path_layers
    ):
        raise SystemExit('--fragment-path-primary requires --fragment-path-layers > 0.')
    if (
        args.encoder_bond_adapter_layers is not None
        and args.encoder_bond_adapter_layers < 0
    ):
        raise SystemExit('--encoder-bond-adapter-layers must be nonnegative.')
    if (
        args.encoder_bond_adapter_feature_dim is not None
        and args.encoder_bond_adapter_feature_dim <= 0
    ):
        raise SystemExit('--encoder-bond-adapter-feature-dim must be positive.')
    if (
        args.encoder_bond_adapter_dropout is not None
        and args.encoder_bond_adapter_dropout < 0
    ):
        raise SystemExit('--encoder-bond-adapter-dropout must be nonnegative.')
    if (
        args.encoder_metadata_adapter_layers is not None
        and args.encoder_metadata_adapter_layers < 0
    ):
        raise SystemExit('--encoder-metadata-adapter-layers must be nonnegative.')
    if (
        args.encoder_metadata_adapter_feature_dim is not None
        and args.encoder_metadata_adapter_feature_dim <= 0
    ):
        raise SystemExit('--encoder-metadata-adapter-feature-dim must be positive.')
    if (
        args.encoder_metadata_adapter_dropout is not None
        and args.encoder_metadata_adapter_dropout < 0
    ):
        raise SystemExit('--encoder-metadata-adapter-dropout must be nonnegative.')
    if (
        args.candidate_transformer_layers is not None
        and args.candidate_transformer_layers < 0
    ):
        raise SystemExit('--candidate-transformer-layers must be nonnegative.')
    if (
        args.candidate_transformer_heads is not None
        and args.candidate_transformer_heads <= 0
    ):
        raise SystemExit('--candidate-transformer-heads must be positive.')
    if (
        args.candidate_transformer_max_tokens is not None
        and args.candidate_transformer_max_tokens <= 0
    ):
        raise SystemExit('--candidate-transformer-max-tokens must be positive.')
    if (
        args.candidate_suppression_gate_hidden_dim is not None
        and args.candidate_suppression_gate_hidden_dim < 0
    ):
        raise SystemExit('--candidate-suppression-gate-hidden-dim must be nonnegative.')
    if (
        args.candidate_suppression_gate_dropout is not None
        and args.candidate_suppression_gate_dropout < 0
    ):
        raise SystemExit('--candidate-suppression-gate-dropout must be nonnegative.')
    if (
        args.candidate_suppression_gate_initial_penalty is not None
        and args.candidate_suppression_gate_initial_penalty <= 0
    ):
        raise SystemExit(
            '--candidate-suppression-gate-initial-penalty must be positive.'
        )
    if args.conditional_expert_heads is not None and args.conditional_expert_heads <= 0:
        raise SystemExit('--conditional-expert-heads must be positive.')
    if (
        args.conditional_expert_hidden_dim is not None
        and args.conditional_expert_hidden_dim <= 0
    ):
        raise SystemExit('--conditional-expert-hidden-dim must be positive.')
    if (
        args.conditional_expert_dropout is not None
        and args.conditional_expert_dropout < 0
    ):
        raise SystemExit('--conditional-expert-dropout must be nonnegative.')
    if (
        args.molecule_descriptor_hidden_dim is not None
        and args.molecule_descriptor_hidden_dim <= 0
    ):
        raise SystemExit('--molecule-descriptor-hidden-dim must be positive.')
    if (
        args.molecule_descriptor_dropout is not None
        and args.molecule_descriptor_dropout < 0
    ):
        raise SystemExit('--molecule-descriptor-dropout must be nonnegative.')
    if args.head_delta_regularization < 0:
        raise SystemExit('--head-delta-regularization must be nonnegative.')
    if args.encoder_delta_regularization < 0:
        raise SystemExit('--encoder-delta-regularization must be nonnegative.')
    if not 0.0 <= args.target_neighbor_weight <= 1.0:
        raise SystemExit('--target-neighbor-weight must be in [0, 1].')
    if args.target_neighbor_ce_window < 0.0:
        raise SystemExit('--target-neighbor-ce-window must be nonnegative.')
    if args.group_balance_power < 0.0:
        raise SystemExit('--group-balance-power must be nonnegative.')
    if args.retrieval_loss_weight < 0:
        raise SystemExit('--retrieval-loss-weight must be nonnegative.')
    if args.distill_loss_weight < 0:
        raise SystemExit('--distill-loss-weight must be nonnegative.')
    if args.distill_min_overlap_mass < 0:
        raise SystemExit('--distill-min-overlap-mass must be nonnegative.')
    if args.distill_oracle_temperature < 0:
        raise SystemExit('--distill-oracle-temperature must be nonnegative.')
    if not 0.0 <= args.distill_oracle_min_teacher_weight < 0.5:
        raise SystemExit('--distill-oracle-min-teacher-weight must be in [0, 0.5).')
    if (
        args.distill_ce_min is not None
        and args.distill_ce_max is not None
        and args.distill_ce_min > args.distill_ce_max
    ):
        raise SystemExit('--distill-ce-min must be <= --distill-ce-max.')
    if args.distill_loss_weight > 0 and not args.distill_predictions:
        raise SystemExit(
            '--distill-predictions is required when --distill-loss-weight > 0.'
        )
    if args.distill_loss_weight > 0 and args.distill_mode == 'rowwise-oracle':
        if len(_parse_comma_list(args.distill_predictions)) != 2:
            raise SystemExit(
                '--distill-mode rowwise-oracle requires exactly two prediction files.'
            )
    if args.retrieval_negatives < 1:
        raise SystemExit('--retrieval-negatives must be positive.')
    if args.retrieval_num_workers is not None and args.retrieval_num_workers < 0:
        raise SystemExit('--retrieval-num-workers must be nonnegative.')
    if (
        args.retrieval_cache_num_workers is not None
        and args.retrieval_cache_num_workers < 0
    ):
        raise SystemExit('--retrieval-cache-num-workers must be nonnegative.')
    if args.retrieval_cache_chunk_size < 1:
        raise SystemExit('--retrieval-cache-chunk-size must be positive.')
    if args.retrieval_sample_timeout < 0:
        raise SystemExit('--retrieval-sample-timeout must be nonnegative.')
    quiet_rdkit_logs()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    df = read_table(args.input)
    if args.massspecgym_filter:
        df = filter_massspecgym_simulation(df)
    if args.max_rows:
        df = df.iloc[: args.max_rows].copy()

    train_df = select_split(
        df,
        split=args.train_split,
        split_col=args.split_col,
        split_value=args.train_split_value,
    )
    val_df = select_split(
        df,
        split=args.val_split,
        split_col=args.split_col,
        split_value=args.val_split_value,
    )
    if train_df.empty:
        raise SystemExit('No training rows selected.')
    if val_df.empty:
        print(
            'Warning: no validation rows selected; using training stats for checkpointing.'
        )

    fine_tune_strategy = args.fine_tune_strategy
    if args.init_checkpoint:
        model, _payload = load_checkpoint(args.init_checkpoint, device=device)
        _validate_loaded_checkpoint_config(
            model,
            mz_max=args.mz_max,
            bin_width=args.bin_width,
        )
        model = _maybe_rebuild_fragment_bond_break_model(model, args, device=device)
        model = _maybe_rebuild_encoder_bond_adapter_model(model, args, device=device)
        model = _maybe_rebuild_encoder_metadata_adapter_model(
            model, args, device=device
        )
        model = _maybe_rebuild_metadata_ce_interaction_model(model, args, device=device)
        _apply_fragment_args_to_model_config(model.config, args)
        _apply_retrieval_calibration_arg(model.config, args)
        model = _maybe_rebuild_retrieval_calibration_model(model, args, device=device)
        set_encoder_finetune_strategy(model, fine_tune_strategy)
        metadata_config = model.metadata_config
        print(
            'Loaded init checkpoint '
            f'{args.init_checkpoint} with encoder_type={model.config.encoder_type} '
            f'fine_tune_strategy={fine_tune_strategy}'
        )
        if args.encoder is not None and args.encoder != model.config.encoder_type:
            print(
                f'Warning: checkpoint encoder_type={model.config.encoder_type!r}; '
                f'ignoring requested --encoder {args.encoder!r}.'
            )
    else:
        encoder_type = args.encoder or 'mace'
        encoder = load_foundation_encoder(
            encoder_type=encoder_type,
            foundation_source=args.foundation_source,
            foundation_model=args.foundation_model,
            foundation_path=args.foundation_path,
            aimnet_model=args.aimnet_model,
            aimnet_path=args.aimnet_path,
            unimol_model_name=args.unimol_model_name,
            unimol_model_size=args.unimol_model_size,
            unimol_pretrained_model_path=args.unimol_pretrained_model_path,
            unimol_pretrained_dict_path=args.unimol_pretrained_dict_path,
            unimol_max_atoms=args.unimol_max_atoms,
            unimol_mode=args.unimol_mode,
            device=device,
        )
        metadata_config = MetadataConfig.from_dataframe(
            train_df,
            precursor_mz_max=args.mz_max,
            collision_energy_max=100.0,
            collision_energy_mode=args.collision_energy_mode,
        )
    metadata_ce_mode = str(
        getattr(metadata_config, 'collision_energy_mode', 'raw') or 'raw'
    ).lower()
    if args.init_checkpoint:
        if (
            args.collision_energy_mode == 'normalized'
            and metadata_ce_mode != 'normalized'
        ):
            metadata_config = replace(
                metadata_config,
                collision_energy_mode='normalized',
            )
            model.metadata_config = metadata_config
            metadata_ce_mode = 'normalized'
            print(
                'Collision-energy metadata override: normalized using loaded '
                'checkpoint CE statistics'
            )
        elif args.collision_energy_mode == 'raw' and metadata_ce_mode != 'raw':
            print(
                'Collision-energy metadata: keeping normalized mode from loaded '
                'checkpoint'
            )
    if metadata_ce_mode == 'normalized':
        train_df = normalize_collision_energy_dataframe(
            train_df, metadata_config=metadata_config
        )
        if not val_df.empty:
            val_df = normalize_collision_energy_dataframe(
                val_df, metadata_config=metadata_config
            )
        print('Collision-energy preprocessing: normalized from training metadata stats')
    if args.merge_group_spectra:
        before_train = len(train_df)
        before_val = len(val_df)
        train_df = merge_group_spectra(
            train_df,
            mz_max=args.mz_max,
            bin_width=args.bin_width,
            group_cols=args.merge_group_cols,
        )
        if not val_df.empty:
            val_df = merge_group_spectra(
                val_df,
                mz_max=args.mz_max,
                bin_width=args.bin_width,
                group_cols=args.merge_group_cols,
            )
        print(
            'Merged group spectra: '
            f'train {before_train}->{len(train_df)} val {before_val}->{len(val_df)}'
        )

    graph_source = model.encoder if args.init_checkpoint else encoder
    graph_config = infer_graph_config(
        graph_source,
        seed=args.seed,
        relaxation=args.graph_relaxation,
        aimnet_relax_model=args.aimnet_relax_model or args.aimnet_model,
        aimnet_relax_steps=args.aimnet_relax_steps,
        aimnet_relax_fmax=args.aimnet_relax_fmax,
        aimnet_relax_device=args.aimnet_relax_device,
    )
    train_df, train_element_stats = filter_supported_elements(
        train_df,
        supported_atomic_numbers=graph_config.atomic_numbers,
    )
    val_df, val_element_stats = filter_supported_elements(
        val_df,
        supported_atomic_numbers=graph_config.atomic_numbers,
    )
    if (
        train_element_stats['dropped_invalid_smiles']
        or train_element_stats['dropped_unsupported_elements']
    ):
        print(f'Training element filter: {train_element_stats}')
    if (
        val_element_stats['dropped_invalid_smiles']
        or val_element_stats['dropped_unsupported_elements']
    ):
        print(f'Validation element filter: {val_element_stats}')
    physical_bond_features = bool(args.physical_bond_features)
    physical_bond_feature_columns = parse_physical_bond_feature_columns(
        args.physical_bond_feature_columns
    )
    if physical_bond_features and not args.physical_bond_features_path:
        raise SystemExit(
            '--physical-bond-features requires --physical-bond-features-path.'
        )

    if train_df.empty:
        raise SystemExit('No training rows left after encoder element filtering.')
    if args.group_balanced_loss:
        train_df = _apply_group_balanced_loss_weights(
            train_df,
            group_cols=args.group_balance_cols,
            power=args.group_balance_power,
        )
    if val_df.empty:
        print(
            'Warning: no validation rows left after encoder element filtering; '
            'using training stats for checkpointing.'
        )
    if not args.init_checkpoint:
        num_bins = num_spectrum_bins(args.mz_max, args.bin_width)
        fragment_action_primary_layers = _mirafrag_config_value(
            args.fragment_action_primary_layers,
            'fragment_action_primary_layers',
        )
        fragment_action_bond_gnn_layers = _mirafrag_config_value(
            args.fragment_action_bond_gnn_layers,
            'fragment_action_bond_gnn_layers',
        )
        bond_break_geometry_features = _new_model_bond_break_geometry_features(
            args,
            action_primary_layers=fragment_action_primary_layers,
            action_bond_gnn_layers=fragment_action_bond_gnn_layers,
        )
        config = MiraFragConfig(
            num_bins=num_bins,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            metadata_dim=args.metadata_dim,
            max_fragment_tree_depth=_mirafrag_config_value(
                args.max_fragment_tree_depth,
                'max_fragment_tree_depth',
            ),
            max_fragment_broken_bonds=_mirafrag_config_value(
                args.max_fragment_broken_bonds,
                'max_fragment_broken_bonds',
            ),
            max_fragments=_mirafrag_config_value(args.max_fragments, 'max_fragments'),
            max_fragment_edges=_mirafrag_config_value(
                args.max_fragment_edges,
                'max_fragment_edges',
            ),
            high_ce_fragment_threshold=args.high_ce_fragment_threshold,
            high_ce_max_fragment_tree_depth=args.high_ce_max_fragment_tree_depth,
            high_ce_max_fragment_broken_bonds=args.high_ce_max_fragment_broken_bonds,
            high_ce_max_fragments=args.high_ce_max_fragments,
            high_ce_max_fragment_edges=args.high_ce_max_fragment_edges,
            include_fragment_isotopes=_mirafrag_config_value(
                args.include_fragment_isotopes,
                'include_fragment_isotopes',
            ),
            fragment_isotope_threshold=_mirafrag_config_value(
                args.fragment_isotope_threshold,
                'fragment_isotope_threshold',
            ),
            max_fragment_isotope_peaks=_mirafrag_config_value(
                args.max_fragment_isotope_peaks,
                'max_fragment_isotope_peaks',
            ),
            fragment_gnn_layers=args.fragment_gnn_layers,
            fragment_path_layers=_mirafrag_config_value(
                args.fragment_path_layers, 'fragment_path_layers'
            ),
            fragment_path_primary=bool(
                _mirafrag_config_value(
                    args.fragment_path_primary, 'fragment_path_primary'
                )
            ),
            fragment_action_path_layers=_mirafrag_config_value(
                args.fragment_action_path_layers,
                'fragment_action_path_layers',
            ),
            fragment_action_primary_layers=fragment_action_primary_layers,
            fragment_action_primary_ce_gate=bool(
                _mirafrag_config_value(
                    args.fragment_action_primary_ce_gate,
                    'fragment_action_primary_ce_gate',
                )
            ),
            bond_break_geometry_features=bond_break_geometry_features,
            fragment_action_geometry_features=bond_break_geometry_features,
            fragment_action_bond_gnn_layers=fragment_action_bond_gnn_layers,
            fragment_bond_break_layers=_mirafrag_config_value(
                args.fragment_bond_break_layers,
                'fragment_bond_break_layers',
            ),
            physical_bond_features=physical_bond_features,
            physical_bond_feature_columns=physical_bond_feature_columns
            if physical_bond_features
            else (),
            fragnnet_dag_layers=_mirafrag_config_value(
                args.fragnnet_dag_layers,
                'fragnnet_dag_layers',
            ),
            fragnnet_dag_num_hs=_mirafrag_config_value(
                args.fragnnet_dag_num_hs,
                'fragnnet_dag_num_hs',
            ),
            ce_basis_features=bool(
                _mirafrag_config_value(args.ce_basis_features, 'ce_basis_features')
            ),
            ce_embedding=_mirafrag_config_value(args.ce_embedding, 'ce_embedding'),
            ce_fourier_frequencies=_mirafrag_config_value(
                args.ce_fourier_frequencies, 'ce_fourier_frequencies'
            ),
            aimnet_charge_features=bool(
                _mirafrag_config_value(
                    args.aimnet_charge_features, 'aimnet_charge_features'
                )
            ),
            aimnet_multipass_features=bool(
                _mirafrag_config_value(
                    args.aimnet_multipass_features, 'aimnet_multipass_features'
                )
            ),
            encoder_bond_adapter_layers=_mirafrag_config_value(
                args.encoder_bond_adapter_layers, 'encoder_bond_adapter_layers'
            ),
            encoder_bond_adapter_feature_dim=_mirafrag_config_value(
                args.encoder_bond_adapter_feature_dim,
                'encoder_bond_adapter_feature_dim',
            ),
            encoder_bond_adapter_dropout=_mirafrag_config_value(
                args.encoder_bond_adapter_dropout, 'encoder_bond_adapter_dropout'
            ),
            encoder_metadata_adapter_layers=_mirafrag_config_value(
                args.encoder_metadata_adapter_layers,
                'encoder_metadata_adapter_layers',
            ),
            encoder_metadata_adapter_feature_dim=_mirafrag_config_value(
                args.encoder_metadata_adapter_feature_dim,
                'encoder_metadata_adapter_feature_dim',
            ),
            encoder_metadata_adapter_dropout=_mirafrag_config_value(
                args.encoder_metadata_adapter_dropout,
                'encoder_metadata_adapter_dropout',
            ),
            metadata_ce_interaction=bool(
                _mirafrag_config_value(
                    args.metadata_ce_interaction, 'metadata_ce_interaction'
                )
            ),
            candidate_transformer_layers=_mirafrag_config_value(
                args.candidate_transformer_layers, 'candidate_transformer_layers'
            ),
            candidate_transformer_heads=_mirafrag_config_value(
                args.candidate_transformer_heads, 'candidate_transformer_heads'
            ),
            candidate_transformer_max_tokens=_mirafrag_config_value(
                args.candidate_transformer_max_tokens,
                'candidate_transformer_max_tokens',
            ),
            candidate_suppression_gate_hidden_dim=_mirafrag_config_value(
                args.candidate_suppression_gate_hidden_dim,
                'candidate_suppression_gate_hidden_dim',
            ),
            candidate_suppression_gate_dropout=_mirafrag_config_value(
                args.candidate_suppression_gate_dropout,
                'candidate_suppression_gate_dropout',
            ),
            candidate_suppression_gate_initial_penalty=_mirafrag_config_value(
                args.candidate_suppression_gate_initial_penalty,
                'candidate_suppression_gate_initial_penalty',
            ),
            conditional_expert_heads=_mirafrag_config_value(
                args.conditional_expert_heads, 'conditional_expert_heads'
            ),
            conditional_expert_hidden_dim=_mirafrag_config_value(
                args.conditional_expert_hidden_dim, 'conditional_expert_hidden_dim'
            ),
            conditional_expert_dropout=_mirafrag_config_value(
                args.conditional_expert_dropout, 'conditional_expert_dropout'
            ),
            molecule_descriptor_features=bool(
                _mirafrag_config_value(
                    args.molecule_descriptor_features,
                    'molecule_descriptor_features',
                )
            ),
            molecule_descriptor_hidden_dim=_mirafrag_config_value(
                args.molecule_descriptor_hidden_dim,
                'molecule_descriptor_hidden_dim',
            ),
            molecule_descriptor_dropout=_mirafrag_config_value(
                args.molecule_descriptor_dropout,
                'molecule_descriptor_dropout',
            ),
            encoder_type=encoder_type,
            encoder_finetune_strategy=fine_tune_strategy,
            foundation_source=args.foundation_source,
            foundation_model=args.foundation_model,
            foundation_path=args.foundation_path,
            aimnet_model=args.aimnet_model,
            aimnet_path=args.aimnet_path,
            unimol_model_name=args.unimol_model_name,
            unimol_model_size=args.unimol_model_size,
            unimol_pretrained_model_path=args.unimol_pretrained_model_path,
            unimol_pretrained_dict_path=args.unimol_pretrained_dict_path,
            unimol_max_atoms=args.unimol_max_atoms,
            unimol_mode=args.unimol_mode,
            retrieval_calibration_head=bool(args.retrieval_calibration_head),
        )
        model = MiraFragModel(encoder, metadata_config=metadata_config, config=config)
    if args.init_checkpoint and args.physical_bond_features is not None:
        model.config.physical_bond_features = physical_bond_features
        model.config.physical_bond_feature_columns = (
            physical_bond_feature_columns if physical_bond_features else ()
        )
    elif args.init_checkpoint and bool(
        getattr(model.config, 'physical_bond_features', False)
    ):
        physical_bond_features = True
        physical_bond_feature_columns = tuple(
            getattr(model.config, 'physical_bond_feature_columns', ())
        )
    if physical_bond_features and not args.physical_bond_features_path:
        raise SystemExit(
            'This checkpoint uses physical bond features; pass --physical-bond-features-path.'
        )
    fragment_support_profile = fragment_support_profile_from_model_config(model.config)

    train_ds = BinnedSpectrumDataset(
        train_df,
        graph_config=graph_config,
        metadata_config=metadata_config,
        mz_max=args.mz_max,
        bin_width=args.bin_width,
        memory_cache=args.memory_cache,
        disk_cache_dir=args.disk_cache_dir,
        include_fragments=True,
        fragment_support_profile=fragment_support_profile,
        slow_sample_seconds=args.slow_sample_seconds,
        trace_samples=args.trace_samples,
        physical_bond_features_path=args.physical_bond_features_path
        if physical_bond_features
        else None,
        physical_bond_feature_columns=physical_bond_feature_columns,
        target_neighbor_smoothing=args.target_neighbor_smoothing,
        target_neighbor_weight=args.target_neighbor_weight,
        target_neighbor_ce_window=args.target_neighbor_ce_window,
        target_neighbor_same_instrument=args.target_neighbor_same_instrument,
        target_neighbor_same_adduct=args.target_neighbor_same_adduct,
        sample_weight_col=SAMPLE_WEIGHT_COLUMN if args.group_balanced_loss else None,
    )
    if args.target_neighbor_smoothing:
        smoothed_rows = sum(
            1 for neighbors in (train_ds._target_neighbor_weights or []) if neighbors
        )
        print(
            'Target neighbor smoothing: '
            f'rows_with_neighbors={smoothed_rows}/{len(train_ds)} '
            f'weight={args.target_neighbor_weight:g} '
            f'ce_window={args.target_neighbor_ce_window:g} '
            f'same_instrument={args.target_neighbor_same_instrument} '
            f'same_adduct={args.target_neighbor_same_adduct}'
        )
    val_ds = (
        BinnedSpectrumDataset(
            val_df,
            graph_config=graph_config,
            metadata_config=metadata_config,
            mz_max=args.mz_max,
            bin_width=args.bin_width,
            memory_cache=args.memory_cache,
            disk_cache_dir=args.disk_cache_dir,
            include_fragments=True,
            fragment_support_profile=fragment_support_profile,
            slow_sample_seconds=args.slow_sample_seconds,
            trace_samples=args.trace_samples,
            physical_bond_features_path=args.physical_bond_features_path
            if physical_bond_features
            else None,
            physical_bond_feature_columns=physical_bond_feature_columns,
        )
        if not val_df.empty
        else None
    )
    if args.disk_cache_dir is not None and args.prefill_cache:
        cache_num_workers = (
            args.num_workers
            if args.cache_num_workers is None
            else args.cache_num_workers
        )
        cache_chunk_size = 1 if args.cache_chunk_size is None else args.cache_chunk_size
        prefill_feature_cache(
            train_ds,
            split_name='train',
            chunk_size=cache_chunk_size,
            num_workers=cache_num_workers,
            show_progress=args.progress,
        )
        if val_ds is not None:
            prefill_feature_cache(
                val_ds,
                split_name='val',
                chunk_size=cache_chunk_size,
                num_workers=cache_num_workers,
                show_progress=args.progress,
            )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_spectrum_batch,
        timeout=args.dataloader_timeout,
        **dataloader_performance_kwargs(
            num_workers=args.num_workers,
            device=device,
        ),
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_spectrum_batch,
            timeout=args.dataloader_timeout,
            **dataloader_performance_kwargs(
                num_workers=args.num_workers,
                device=device,
            ),
        )
        if val_ds is not None
        else None
    )
    retrieval_loader = _build_retrieval_loader(
        args,
        train_df=train_df,
        graph_config=graph_config,
        metadata_config=metadata_config,
        fragment_support_profile=fragment_support_profile,
        device=device,
    )
    distill_spectra = _load_distillation_spectra(
        args,
        train_df=train_df,
        mz_max=args.mz_max,
        bin_width=args.bin_width,
    )

    if args.validation_tune:
        del model
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
        _run_validation_tuning(
            args,
            train_loader,
            val_loader,
            device=device,
            fine_tune_strategy=fine_tune_strategy,
            graph_config=graph_config,
            metadata_config=metadata_config,
        )
        return

    train_model(
        model,
        train_loader,
        val_loader,
        retrieval_loader=retrieval_loader,
        retrieval_loss_weight=args.retrieval_loss_weight,
        distill_spectra=distill_spectra,
        distill_loss_weight=args.distill_loss_weight,
        distill_min_overlap_mass=args.distill_min_overlap_mass,
        epochs=args.epochs,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        head_lr=args.head_learning_rate,
        encoder_lr=args.encoder_learning_rate,
        head_weight_decay=args.head_weight_decay,
        encoder_weight_decay=args.encoder_weight_decay,
        encoder_layer_lr_decay=args.encoder_layer_lr_decay,
        head_delta_regularization=args.head_delta_regularization,
        encoder_delta_regularization=args.encoder_delta_regularization,
        output=args.output,
        loss_name=args.loss,
        train_config=_train_config(args, fine_tune_strategy=fine_tune_strategy),
        graph_config=graph_config,
        show_progress=args.progress,
        scheduler_name=args.scheduler,
        scheduler_interval=args.scheduler_interval,
        min_lr_ratio=args.min_lr_ratio,
        exponential_gamma=args.exponential_gamma,
        plateau_factor=args.plateau_factor,
        plateau_patience=args.plateau_patience,
        evaluate_initial=bool(args.init_checkpoint),
        mass_tolerance=args.mass_tolerance,
        relative_mass_tolerance=args.relative_mass_tolerance,
        mass_tolerance_min_mz=args.mass_tolerance_min_mz,
        kl_weight=args.kl_weight,
        coverage_weight=args.coverage_weight,
        target_power=args.target_power,
        entropy_weight=args.entropy_weight,
        checkpoint_metric=args.checkpoint_metric,
        verbose_epoch_config=False,
        swa=args.swa,
        swa_start_epoch=args.swa_start_epoch,
        swa_lr=args.swa_lr,
        swa_anneal_epochs=args.swa_anneal_epochs,
    )


def _load_distillation_spectra(
    args: argparse.Namespace,
    *,
    train_df: pd.DataFrame,
    mz_max: float,
    bin_width: float,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]] | None:
    if args.distill_loss_weight <= 0.0:
        return None
    paths = _parse_comma_list(args.distill_predictions)
    if not paths:
        raise SystemExit('--distill-predictions must contain at least one path.')
    tables = [pd.read_csv(path) for path in paths]
    lookups = [
        _prediction_spectrum_lookup(
            table,
            path,
            mz_max=mz_max,
            bin_width=bin_width,
        )
        for table, path in zip(tables, paths)
    ]
    identifiers = set(lookups[0])
    for lookup in lookups[1:]:
        identifiers &= set(lookup)
    identifiers &= _distillation_allowed_identifiers(
        train_df,
        ce_min=args.distill_ce_min,
        ce_max=args.distill_ce_max,
    )
    if args.distill_mode == 'rowwise-oracle':
        weights_by_identifier = _rowwise_oracle_distill_weights(
            train_df,
            lookups,
            identifiers=identifiers,
            mz_max=mz_max,
            bin_width=bin_width,
            temperature=args.distill_oracle_temperature,
            min_teacher_weight=args.distill_oracle_min_teacher_weight,
        )
        identifiers &= set(weights_by_identifier)
        fixed_weights = None
    else:
        fixed_weights = _parse_distill_weights(args.distill_weights, n_paths=len(paths))
        weights_by_identifier = None
    if not identifiers:
        raise SystemExit('Distillation prediction files have no shared identifiers.')
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    missing = 0
    for identifier in sorted(identifiers):
        spectra = []
        weights = (
            weights_by_identifier[identifier]
            if weights_by_identifier is not None
            else fixed_weights
        )
        if weights is None:
            raise AssertionError('distillation weights were not initialized')
        for lookup, weight in zip(lookups, weights):
            if float(weight) <= 0.0:
                continue
            spectrum = lookup.get(identifier)
            if spectrum is None:
                missing += 1
                continue
            spectra.append(scale_sparse_spectrum(spectrum, weight))
        if not spectra:
            continue
        teacher = combine_sparse_spectra(spectra)
        out[identifier] = (
            torch.as_tensor(teacher.bins, dtype=torch.long),
            torch.as_tensor(teacher.values, dtype=torch.float32),
        )
    if not out:
        raise SystemExit('Distillation teacher spectra are empty.')
    print(
        'Loaded distillation teacher spectra: '
        f'rows={len(out)} files={len(paths)} '
        + _distillation_weight_summary(
            args,
            fixed_weights=fixed_weights,
            weights_by_identifier=weights_by_identifier,
        )
        + _distillation_filter_summary(args)
        + (f' missing_entries={missing}' if missing else '')
    )
    return out


def _rowwise_oracle_distill_weights(
    train_df: pd.DataFrame,
    lookups: list[dict[str, SparseSpectrum]],
    *,
    identifiers: set[str],
    mz_max: float,
    bin_width: float,
    temperature: float,
    min_teacher_weight: float,
) -> dict[str, tuple[float, float]]:
    if len(lookups) != 2:
        raise ValueError('rowwise oracle distillation requires exactly two teachers.')
    targets = _distillation_target_lookup(train_df, mz_max=mz_max, bin_width=bin_width)
    out: dict[str, tuple[float, float]] = {}
    first_wins = 0
    second_wins = 0
    total_first_weight = 0.0
    total_first_cosine = 0.0
    total_second_cosine = 0.0
    total_oracle_cosine = 0.0
    for identifier in sorted(identifiers):
        target = targets.get(identifier)
        first = lookups[0].get(identifier)
        second = lookups[1].get(identifier)
        if target is None or first is None or second is None:
            continue
        first_cosine = sparse_cosine(first, target)
        second_cosine = sparse_cosine(second, target)
        if temperature == 0.0:
            first_weight = 1.0 if first_cosine >= second_cosine else 0.0
        else:
            margin = (first_cosine - second_cosine) / float(temperature)
            first_weight = 1.0 / (1.0 + math.exp(-max(min(margin, 60.0), -60.0)))
        if min_teacher_weight > 0.0:
            lo = float(min_teacher_weight)
            first_weight = min(max(first_weight, lo), 1.0 - lo)
        second_weight = 1.0 - first_weight
        out[identifier] = (float(first_weight), float(second_weight))
        first_wins += int(first_cosine >= second_cosine)
        second_wins += int(second_cosine > first_cosine)
        total_first_weight += float(first_weight)
        total_first_cosine += float(first_cosine)
        total_second_cosine += float(second_cosine)
        total_oracle_cosine += max(float(first_cosine), float(second_cosine))
    if not out:
        raise SystemExit('Rowwise-oracle distillation found no scoreable teacher rows.')
    n = len(out)
    print(
        'Rowwise-oracle distillation weights: '
        f'rows={n} first_wins={first_wins} second_wins={second_wins} '
        f'first_weight_mean={total_first_weight / n:.4f} '
        f'first_cosine_mean={total_first_cosine / n:.5f} '
        f'second_cosine_mean={total_second_cosine / n:.5f} '
        f'oracle_cosine_mean={total_oracle_cosine / n:.5f} '
        f'temperature={float(temperature):g} '
        f'min_teacher_weight={float(min_teacher_weight):g}'
    )
    return out


def _distillation_target_lookup(
    train_df: pd.DataFrame,
    *,
    mz_max: float,
    bin_width: float,
) -> dict[str, SparseSpectrum]:
    identifier_col = find_column(train_df, ('identifier',), required=False)
    precursor_col = find_column(train_df, PRECURSOR_ALIASES, required=False)
    out: dict[str, SparseSpectrum] = {}
    for idx, row in train_df.iterrows():
        identifier = (
            str(row[identifier_col]) if identifier_col is not None else str(idx)
        )
        precursor_mz = row.get(precursor_col) if precursor_col is not None else None
        mzs, intensities = parse_peaks(
            row,
            precursor_mz=precursor_mz,
            exclude_precursor=True,
            precursor_tolerance=bin_width,
        )
        out[identifier] = sparse_from_peaks(
            mzs,
            intensities,
            mz_max=mz_max,
            bin_width=bin_width,
        )
    return out


def _distillation_weight_summary(
    args: argparse.Namespace,
    *,
    fixed_weights: list[float] | None,
    weights_by_identifier: dict[str, tuple[float, float]] | None,
) -> str:
    if args.distill_mode == 'rowwise-oracle':
        if not weights_by_identifier:
            return 'mode=rowwise-oracle'
        first_mean = sum(w[0] for w in weights_by_identifier.values()) / len(
            weights_by_identifier
        )
        return f'mode=rowwise-oracle first_weight_mean={first_mean:.4f}'
    if fixed_weights is None:
        raise AssertionError('fixed distillation weights were not initialized')
    return 'weights=' + ':'.join(f'{w:g}' for w in fixed_weights)


def _distillation_allowed_identifiers(
    train_df: pd.DataFrame,
    *,
    ce_min: float | None,
    ce_max: float | None,
) -> set[str]:
    identifier_col = find_column(train_df, ('identifier',), required=False)
    identifiers = (
        train_df.index.astype(str)
        if identifier_col is None
        else train_df[identifier_col].astype(str)
    )
    if ce_min is None and ce_max is None:
        return set(identifiers)
    ce_col = (
        RAW_COLLISION_ENERGY_COLUMN
        if RAW_COLLISION_ENERGY_COLUMN in train_df.columns
        else find_column(train_df, CE_ALIASES, required=False)
    )
    if ce_col is None:
        raise SystemExit(
            '--distill-ce-min/--distill-ce-max require a collision-energy column.'
        )
    ce_values = pd.to_numeric(train_df[ce_col], errors='coerce')
    mask = ce_values.notna()
    if ce_min is not None:
        mask &= ce_values >= float(ce_min)
    if ce_max is not None:
        mask &= ce_values <= float(ce_max)
    return set(identifiers[mask])


def _distillation_filter_summary(args: argparse.Namespace) -> str:
    parts = []
    if args.distill_ce_min is not None:
        parts.append(f'ce_min={float(args.distill_ce_min):g}')
    if args.distill_ce_max is not None:
        parts.append(f'ce_max={float(args.distill_ce_max):g}')
    return f' filter={",".join(parts)}' if parts else ''


def _prediction_spectrum_lookup(
    table: pd.DataFrame,
    path: str,
    *,
    mz_max: float,
    bin_width: float,
) -> dict[str, Any]:
    if 'identifier' not in table.columns or 'pred_peaks' not in table.columns:
        raise SystemExit(
            f'Distillation prediction file {path!r} requires identifier and pred_peaks columns.'
        )
    lookup = {}
    for row in table.itertuples(index=False):
        identifier = str(getattr(row, 'identifier'))
        peaks = getattr(row, 'pred_peaks')
        if isinstance(peaks, str):
            peaks = json.loads(peaks)
        lookup[identifier] = sparse_from_peaks(
            peaks.get('mz', peaks.get('mzs', [])),
            peaks.get('intensity', peaks.get('intensities', [])),
            mz_max=mz_max,
            bin_width=bin_width,
        )
    return lookup


def _parse_comma_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [
        part.strip() for part in str(value).replace(';', ',').split(',') if part.strip()
    ]


def _parse_distill_weights(value: str | None, *, n_paths: int) -> list[float]:
    if n_paths <= 0:
        raise ValueError('n_paths must be positive.')
    if value is None or not str(value).strip():
        return [1.0 / n_paths for _ in range(n_paths)]
    parts = [
        part.strip() for part in str(value).replace(':', ',').split(',') if part.strip()
    ]
    weights = [float(part) for part in parts]
    if len(weights) != n_paths:
        raise SystemExit('--distill-weights must match --distill-predictions length.')
    total = sum(weights)
    if total <= 0.0:
        raise SystemExit('--distill-weights must sum to a positive value.')
    return [weight / total for weight in weights]


def _maybe_rebuild_metadata_ce_interaction_model(
    model: MiraFragModel,
    args: argparse.Namespace,
    *,
    device: str | torch.device,
) -> MiraFragModel:
    """Rebuild checkpoint model when metadata CE interaction changes."""
    requested = getattr(args, 'metadata_ce_interaction', None)
    target = (
        bool(getattr(model.config, 'metadata_ce_interaction', False))
        if requested is None
        else bool(requested)
    )
    current = bool(getattr(model.config, 'metadata_ce_interaction', False))
    has_adapter = getattr(model, 'metadata_ce_interaction', None) is not None
    if target == current and target == has_adapter:
        return model
    config = replace(model.config, metadata_ce_interaction=target)
    rebuilt = MiraFragModel(
        model.encoder,
        metadata_config=model.metadata_config,
        config=config,
    ).to(device)
    incompatible = rebuilt.load_state_dict(model.state_dict(), strict=False)
    allowed_prefix = 'metadata_ce_interaction.'
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith(allowed_prefix)
    ]
    missing = [
        key for key in incompatible.missing_keys if not key.startswith(allowed_prefix)
    ]
    if missing or unexpected:
        raise RuntimeError(
            'Unexpected checkpoint mismatch while changing metadata CE interaction: '
            f'missing={missing} unexpected={unexpected}'
        )
    branch_mismatch = len(incompatible.missing_keys) + len(incompatible.unexpected_keys)
    if branch_mismatch:
        print('Loaded checkpoint with metadata CE interaction initialized as no-op.')
    return rebuilt


def _apply_retrieval_calibration_arg(
    config: MiraFragConfig, args: argparse.Namespace
) -> None:
    if args.retrieval_calibration_head is not None:
        config.retrieval_calibration_head = bool(args.retrieval_calibration_head)


def _maybe_rebuild_retrieval_calibration_model(
    model: MiraFragModel,
    args: argparse.Namespace,
    *,
    device: str | torch.device,
) -> MiraFragModel:
    enabled = bool(getattr(model.config, 'retrieval_calibration_head', False))
    has_head = getattr(model, 'retrieval_calibration_head', None) is not None
    if enabled == has_head:
        return model
    rebuilt = MiraFragModel(
        model.encoder,
        metadata_config=model.metadata_config,
        config=model.config,
    ).to(device)
    incompatible = rebuilt.load_state_dict(model.state_dict(), strict=False)
    allowed_missing = 'retrieval_calibration_head.'
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith(allowed_missing)
    ]
    missing = [
        key for key in incompatible.missing_keys if not key.startswith(allowed_missing)
    ]
    if missing or unexpected:
        raise RuntimeError(
            'Failed to rebuild retrieval calibration model: '
            f'missing={missing} unexpected={unexpected}'
        )
    print('Loaded checkpoint with retrieval calibration head initialized as no-op.')
    return rebuilt


def _build_retrieval_loader(
    args: argparse.Namespace,
    *,
    train_df,
    graph_config,
    metadata_config: MetadataConfig,
    fragment_support_profile,
    device: str | torch.device,
) -> DataLoader | None:
    if args.retrieval_loss_weight <= 0.0:
        return None
    if not args.retrieval_candidate_input:
        raise SystemExit(
            '--retrieval-candidate-input is required when --retrieval-loss-weight > 0.'
        )
    query_df = train_df
    if args.retrieval_max_queries is not None:
        query_df = query_df.iloc[: int(args.retrieval_max_queries)].copy()
    candidate_pool = read_retrieval_candidate_table(args.retrieval_candidate_input)
    retrieval_df = build_retrieval_candidate_rows(
        query_df,
        candidate_pool,
        mode='explicit',
        max_candidates=int(args.retrieval_negatives) + 1,
        seed=int(args.seed),
        ensure_true_candidate=True,
    )
    if retrieval_df.empty:
        raise SystemExit('Auxiliary retrieval training produced no candidate rows.')
    retrieval_df, element_stats = filter_supported_elements(
        retrieval_df,
        supported_atomic_numbers=graph_config.atomic_numbers,
    )
    if (
        element_stats['dropped_invalid_smiles']
        or element_stats['dropped_unsupported_elements']
    ):
        print(f'Retrieval auxiliary element filter: {element_stats}')
    if retrieval_df.empty:
        raise SystemExit(
            'Auxiliary retrieval training produced no supported candidate rows.'
        )
    group_sizes = retrieval_df.groupby('_retrieval_query_identifier', sort=False).size()
    multi_candidate_groups = set(group_sizes[group_sizes > 1].index)
    if len(multi_candidate_groups) < len(group_sizes):
        retrieval_df = retrieval_df[
            retrieval_df['_retrieval_query_identifier'].isin(multi_candidate_groups)
        ].copy()
    if retrieval_df.empty:
        raise SystemExit(
            'Auxiliary retrieval training produced no multi-candidate query groups.'
        )
    batches = _retrieval_group_batches(
        retrieval_df,
        max_batch_size=int(args.batch_size),
        seed=int(args.seed),
    )
    print(
        'Retrieval auxiliary training: '
        f'queries={retrieval_df["_retrieval_query_identifier"].nunique()} '
        f'candidate_rows={len(retrieval_df)} '
        f'batches={len(batches)} '
        f'loss_weight={args.retrieval_loss_weight:g}'
    )
    retrieval_num_workers = (
        int(args.num_workers)
        if args.retrieval_num_workers is None
        else int(args.retrieval_num_workers)
    )
    dataset = BinnedSpectrumDataset(
        retrieval_df,
        graph_config=graph_config,
        metadata_config=metadata_config,
        mz_max=args.mz_max,
        bin_width=args.bin_width,
        memory_cache=args.memory_cache,
        disk_cache_dir=args.disk_cache_dir,
        include_fragments=True,
        fragment_support_profile=fragment_support_profile,
        slow_sample_seconds=args.slow_sample_seconds,
        trace_samples=args.trace_samples,
    )
    if args.disk_cache_dir is not None and args.retrieval_prefill_cache:
        cache_df = _retrieval_cache_rows(retrieval_df)
        cache_dataset = _SafeAuxRetrievalDataset(
            BinnedSpectrumDataset(
                cache_df,
                graph_config=graph_config,
                metadata_config=metadata_config,
                mz_max=args.mz_max,
                bin_width=args.bin_width,
                require_spectrum=False,
                memory_cache=args.memory_cache,
                disk_cache_dir=args.disk_cache_dir,
                include_fragments=True,
                fragment_support_profile=fragment_support_profile,
                slow_sample_seconds=args.slow_sample_seconds,
                trace_samples=args.trace_samples,
            ),
            timeout_seconds=float(args.retrieval_sample_timeout),
        )
        retrieval_cache_workers = (
            retrieval_num_workers
            if args.retrieval_cache_num_workers is None
            else int(args.retrieval_cache_num_workers)
        )
        failures = prefill_feature_cache(
            cache_dataset,
            split_name='retrieval train',
            chunk_size=int(args.retrieval_cache_chunk_size),
            num_workers=retrieval_cache_workers,
            show_progress=args.progress,
            ignore_errors=True,
        )
        if failures:
            print(
                'Retrieval auxiliary cache prefill skipped '
                f'{len(failures)} uncacheable candidate rows.'
            )
    return DataLoader(
        _SafeAuxRetrievalDataset(
            dataset,
            timeout_seconds=float(args.retrieval_sample_timeout),
        ),
        batch_sampler=batches,
        num_workers=retrieval_num_workers,
        collate_fn=_collate_aux_retrieval_batch,
        timeout=args.dataloader_timeout,
        **dataloader_performance_kwargs(
            num_workers=retrieval_num_workers,
            device=device,
        ),
    )


def _resolve_group_balance_cols(
    df: pd.DataFrame,
    group_cols: str | list[str] | tuple[str, ...],
) -> list[str]:
    if group_cols != 'auto':
        if isinstance(group_cols, str):
            cols = [col.strip() for col in group_cols.split(',') if col.strip()]
        else:
            cols = [str(col) for col in group_cols]
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(f'Group-balance columns are missing: {missing}')
        if not cols:
            raise ValueError('At least one group-balance column is required.')
        return cols
    cols: list[str] = []
    smiles_col = find_column(df, SMILES_ALIASES, required=False)
    adduct_col = find_column(df, ADDUCT_ALIASES, required=False)
    instrument_col = find_column(df, INSTRUMENT_ALIASES, required=False)
    for col in (smiles_col, adduct_col, instrument_col):
        if col is not None and col not in cols:
            cols.append(col)
    if not cols:
        raise ValueError('Could not infer group-balance columns.')
    return cols


def _apply_group_balanced_loss_weights(
    df: pd.DataFrame,
    *,
    group_cols: str | list[str] | tuple[str, ...],
    power: float = 1.0,
) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    cols = _resolve_group_balance_cols(out, group_cols)
    group_ids = out.groupby(cols, dropna=False, sort=False).ngroup()
    counts_by_group = pd.Series(group_ids).value_counts(sort=False)
    counts = pd.Series(group_ids).map(counts_by_group).astype(float)
    raw_weights = counts.pow(-float(power))
    mean_weight = float(raw_weights.mean()) if len(raw_weights) else 1.0
    if not math.isfinite(mean_weight) or mean_weight <= 0.0:
        raise ValueError('Could not normalize group-balanced loss weights.')
    weights = raw_weights / mean_weight
    out[SAMPLE_WEIGHT_COLUMN] = weights.to_numpy(dtype='float32')
    print(
        'Group-balanced loss weights: '
        f'rows={len(out)} groups={int(counts_by_group.size)} '
        f'cols={",".join(cols)} power={float(power):g} '
        f'min={float(weights.min()):.4g} max={float(weights.max()):.4g} '
        f'mean={float(weights.mean()):.4g}'
    )
    return out


def _retrieval_cache_rows(retrieval_df):
    key_cols = []
    for aliases in (SMILES_ALIASES, ADDUCT_ALIASES, CE_ALIASES):
        column = find_column(retrieval_df, aliases, required=False)
        if column is not None and column not in key_cols:
            key_cols.append(column)
    if RAW_COLLISION_ENERGY_COLUMN in retrieval_df.columns:
        key_cols.append(RAW_COLLISION_ENERGY_COLUMN)
    if not key_cols:
        return retrieval_df.reset_index(drop=True)
    return retrieval_df.drop_duplicates(subset=key_cols).reset_index(drop=True)


class _SafeAuxRetrievalDataset:
    def __init__(
        self,
        dataset: BinnedSpectrumDataset,
        *,
        timeout_seconds: float = 0.0,
    ):
        self.dataset = dataset
        self.df = getattr(dataset, 'df', None)
        self.smiles_col = getattr(dataset, 'smiles_col', None)
        self.adduct_col = getattr(dataset, 'adduct_col', None)
        self.instrument_col = getattr(dataset, 'instrument_col', None)
        self.ce_col = getattr(dataset, 'ce_col', None)
        self.timeout_seconds = max(0.0, float(timeout_seconds))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        try:
            item = self._run_with_timeout(lambda: self.dataset[int(idx)])
        except Exception as exc:  # noqa: BLE001
            return {
                '_retrieval_failed': True,
                '_retrieval_error': str(exc),
            }
        item['_retrieval_failed'] = False
        return item

    def materialize_feature_cache(self, idx: int) -> None:
        def materialize() -> None:
            materialize_cache = getattr(self.dataset, 'materialize_feature_cache', None)
            if callable(materialize_cache):
                materialize_cache(int(idx))
            else:
                self.dataset[int(idx)]

        self._run_with_timeout(materialize)

    def _run_with_timeout(self, fn):
        previous_handler = None
        timeout_active = self.timeout_seconds > 0.0
        try:
            if timeout_active:
                previous_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, _aux_retrieval_timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, self.timeout_seconds)
            return fn()
        finally:
            if timeout_active:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, previous_handler)


def _aux_retrieval_timeout_handler(_signum, _frame) -> None:
    raise TimeoutError('auxiliary retrieval candidate feature generation timed out')


def _collate_aux_retrieval_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    good = [item for item in items if not item.get('_retrieval_failed')]
    failed = [item for item in items if item.get('_retrieval_failed')]
    if not good:
        return {
            '_retrieval_empty': True,
            '_retrieval_errors': [
                str(item.get('_retrieval_error', '')) for item in failed
            ],
        }
    batch = collate_spectrum_batch(good)
    batch['_retrieval_empty'] = False
    batch['_retrieval_errors'] = [
        str(item.get('_retrieval_error', '')) for item in failed
    ]
    return batch


def _retrieval_group_batches(
    retrieval_df,
    *,
    max_batch_size: int,
    seed: int,
) -> list[list[int]]:
    if max_batch_size < 1:
        raise ValueError('retrieval batch size must be positive.')
    groups = [
        list(map(int, positions))
        for positions in retrieval_df.groupby(
            '_retrieval_query_identifier',
            sort=False,
        ).indices.values()
    ]
    generator = torch.Generator()
    generator.manual_seed(seed)
    order = torch.randperm(len(groups), generator=generator).tolist()
    batches: list[list[int]] = []
    current: list[int] = []
    for group_idx in order:
        group = groups[int(group_idx)]
        if current and len(current) + len(group) > max_batch_size:
            batches.append(current)
            current = []
        if len(group) > max_batch_size:
            batches.append(group)
        else:
            current.extend(group)
    if current:
        batches.append(current)
    return batches


def _run_validation_tuning(
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    device: str | torch.device,
    fine_tune_strategy: str,
    graph_config,
    metadata_config: MetadataConfig,
) -> None:
    """
    Run short validation-driven fine-tuning trials from one checkpoint.

    The initial checkpoint is evaluated once as the reference score. Each trial
    then reloads ``--init-checkpoint`` so optimizer state, model weights,
    dropout probability, and SWA settings are independent. The best checkpoint
    by ``--tune-metric`` is copied to ``--output``.
    """
    if args.init_checkpoint is None:
        raise SystemExit('--validation-tune requires --init-checkpoint.')
    if val_loader is None:
        raise SystemExit('--validation-tune requires a non-empty validation split.')
    if args.tune_epochs < 1:
        raise SystemExit('--tune-epochs must be positive.')

    candidates = _validation_tune_candidates(args)
    output = Path(args.output)
    trial_dir = _validation_tune_trial_dir(args, output)
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial_summaries: list[dict[str, Any]] = []

    print(
        'validation tuning enabled: '
        f'trials={len(candidates)} metric={args.tune_metric} '
        f'epochs_per_trial={args.tune_epochs} trial_dir={trial_dir}'
    )

    initial_output = trial_dir / f'{output.stem}.initial.pt'
    initial_stats = _evaluate_validation_tune_initial(
        args,
        val_loader,
        device=device,
        fine_tune_strategy=fine_tune_strategy,
        metadata_config=metadata_config,
        graph_config=graph_config,
        output=initial_output,
    )
    best_score = _tune_score_from_val_stats(initial_stats, args.tune_metric)
    best_trial_output: Path | None = initial_output
    best_trial_history: Path | None = None
    best_summary: dict[str, Any] | None = {
        'trial': 0,
        'score': best_score,
        'metric': args.tune_metric,
        'checkpoint': str(best_trial_output),
        'history': str(best_trial_history) if best_trial_history is not None else None,
        'candidate': None,
        'initial': True,
    }
    print(
        'validation tuning initial: '
        f'val_loss={initial_stats["loss"]:.5f} '
        f'val_cosine={initial_stats["cosine"]:.5f} '
        f'val_oos={initial_stats["oos_probability"]:.5f} '
        f'{args.tune_metric}={best_score:.5f}'
    )

    for trial_index, candidate in enumerate(candidates, start=1):
        trial_output = trial_dir / f'{output.stem}.trial{trial_index:03d}.pt'
        print(
            'validation tuning trial '
            f'{trial_index}/{len(candidates)} '
            f'head_lr={candidate.head_lr:.2e} '
            f'encoder_lr={candidate.encoder_lr:.2e} '
            f'dropout={candidate.dropout:g} '
            f'head_weight_decay={candidate.head_weight_decay:.2e} '
            f'encoder_weight_decay={candidate.encoder_weight_decay:.2e} '
            f'swa={candidate.swa} '
            f'swa_start={candidate.swa_start_epoch} '
            f'swa_lr={candidate.swa_lr}'
        )
        trial_model, _payload = load_checkpoint(args.init_checkpoint, device=device)
        _validate_loaded_checkpoint_config(
            trial_model,
            mz_max=args.mz_max,
            bin_width=args.bin_width,
        )
        trial_model = _maybe_rebuild_fragment_bond_break_model(
            trial_model, args, device=device
        )
        trial_model = _maybe_rebuild_encoder_bond_adapter_model(
            trial_model, args, device=device
        )
        trial_model = _maybe_rebuild_encoder_metadata_adapter_model(
            trial_model, args, device=device
        )
        trial_model = _maybe_rebuild_metadata_ce_interaction_model(
            trial_model, args, device=device
        )
        _apply_fragment_args_to_model_config(trial_model.config, args)
        trial_model.metadata_config = metadata_config
        set_encoder_finetune_strategy(trial_model, fine_tune_strategy)
        _set_head_dropout(trial_model, candidate.dropout)
        trial_train_config = _train_config(
            args,
            fine_tune_strategy=fine_tune_strategy,
        )
        trial_train_config['validation_tune_trial'] = trial_index
        trial_train_config['validation_tune_metric'] = args.tune_metric
        trial_train_config['validation_tune_candidate'] = asdict(candidate)

        history = train_model(
            trial_model,
            train_loader,
            val_loader,
            epochs=args.tune_epochs,
            lr=candidate.head_lr,
            weight_decay=candidate.encoder_weight_decay,
            device=device,
            head_lr=candidate.head_lr,
            encoder_lr=candidate.encoder_lr,
            head_weight_decay=candidate.head_weight_decay,
            encoder_weight_decay=candidate.encoder_weight_decay,
            encoder_layer_lr_decay=args.encoder_layer_lr_decay,
            head_delta_regularization=args.head_delta_regularization,
            encoder_delta_regularization=args.encoder_delta_regularization,
            output=trial_output,
            loss_name=args.loss,
            train_config=trial_train_config,
            graph_config=graph_config,
            show_progress=args.progress,
            scheduler_name=args.scheduler,
            scheduler_interval=args.scheduler_interval,
            min_lr_ratio=args.min_lr_ratio,
            exponential_gamma=args.exponential_gamma,
            plateau_factor=args.plateau_factor,
            plateau_patience=args.plateau_patience,
            evaluate_initial=False,
            mass_tolerance=args.mass_tolerance,
            relative_mass_tolerance=args.relative_mass_tolerance,
            mass_tolerance_min_mz=args.mass_tolerance_min_mz,
            kl_weight=args.kl_weight,
            coverage_weight=args.coverage_weight,
            target_power=args.target_power,
            entropy_weight=args.entropy_weight,
            checkpoint_metric=args.tune_metric,
            verbose_epoch_config=True,
            swa=candidate.swa,
            swa_start_epoch=candidate.swa_start_epoch,
            swa_lr=candidate.swa_lr,
            swa_anneal_epochs=candidate.swa_anneal_epochs,
        )
        trial_score = _best_tune_score(history, args.tune_metric)
        trial_summary = {
            'trial': trial_index,
            'score': trial_score,
            'metric': args.tune_metric,
            'checkpoint': str(trial_output),
            'history': str(_history_path(trial_output)),
            'candidate': asdict(candidate),
        }
        trial_summaries.append(trial_summary)
        print(f'validation tuning trial complete: {args.tune_metric}={trial_score:.5f}')
        if _tune_improved(trial_score, best_score, args.tune_metric):
            best_score = trial_score
            best_trial_output = trial_output
            best_trial_history = _history_path(trial_output)
            best_summary = trial_summary
            print(
                'validation tuning new best: '
                f'trial={trial_index} {args.tune_metric}={best_score:.5f}'
            )

    if best_trial_output is None or best_summary is None:
        raise RuntimeError('Validation tuning did not produce a checkpoint.')
    output.parent.mkdir(parents=True, exist_ok=True)
    if best_trial_output.resolve() != output.resolve():
        shutil.copy2(best_trial_output, output)
    if best_trial_history is not None and best_trial_history.exists():
        output_history = _history_path(output)
        if best_trial_history.resolve() != output_history.resolve():
            shutil.copy2(best_trial_history, output_history)
    summary_path = _tune_summary_path(output)
    with open(summary_path, 'w') as fp:
        json.dump(
            {
                'metric': args.tune_metric,
                'best_score': best_score,
                'best_trial': best_summary,
                'trials': trial_summaries,
            },
            fp,
            indent=2,
        )
    print(
        'validation tuning best checkpoint copied: '
        f'{best_trial_output} -> {output} '
        f'{args.tune_metric}={best_score:.5f}'
    )
    if not args.tune_keep_trials:
        _cleanup_validation_tune_trials(trial_summaries, trial_dir)


def _evaluate_validation_tune_initial(
    args: argparse.Namespace,
    val_loader: DataLoader,
    *,
    device: str | torch.device,
    fine_tune_strategy: str,
    metadata_config: MetadataConfig,
    graph_config,
    output: Path,
) -> dict[str, float]:
    """
    Evaluate the initial validation-tuning checkpoint without training it.
    """
    model, _payload = load_checkpoint(args.init_checkpoint, device=device)
    _validate_loaded_checkpoint_config(
        model,
        mz_max=args.mz_max,
        bin_width=args.bin_width,
    )
    model = _maybe_rebuild_fragment_bond_break_model(model, args, device=device)
    model = _maybe_rebuild_encoder_bond_adapter_model(model, args, device=device)
    model = _maybe_rebuild_encoder_metadata_adapter_model(model, args, device=device)
    model = _maybe_rebuild_metadata_ce_interaction_model(model, args, device=device)
    _apply_fragment_args_to_model_config(model.config, args)
    model.metadata_config = metadata_config
    set_encoder_finetune_strategy(model, fine_tune_strategy)
    model.to(device)
    stats = run_epoch(
        model,
        val_loader,
        optimizer=None,
        device=device,
        loss_name=args.loss,
        desc=f'val initial/{args.tune_epochs}',
        show_progress=args.progress,
        mass_tolerance=args.mass_tolerance,
        relative_mass_tolerance=args.relative_mass_tolerance,
        mass_tolerance_min_mz=args.mass_tolerance_min_mz,
        kl_weight=args.kl_weight,
        coverage_weight=args.coverage_weight,
        target_power=args.target_power,
        entropy_weight=args.entropy_weight,
    )
    save_checkpoint(
        output,
        model,
        train_config={
            **_train_config(args, fine_tune_strategy=fine_tune_strategy),
            'validation_tune_trial': 0,
            'validation_tune_metric': args.tune_metric,
            'initial': True,
        },
        graph_config=graph_config,
    )
    del model
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return stats


def _validation_tune_candidates(
    args: argparse.Namespace,
) -> list[_ValidationTuneCandidate]:
    """
    Build a deterministic, shuffled list of requested validation-tuning candidates.
    """
    if args.tune_trials < 1:
        raise SystemExit('--tune-trials must be positive.')
    tune_epochs = int(args.tune_epochs)
    tune_head_lrs = getattr(args, 'tune_head_lrs', None)
    tune_encoder_lrs = getattr(args, 'tune_encoder_lrs', None)
    tune_head_weight_decays = getattr(args, 'tune_head_weight_decays', None)
    tune_encoder_weight_decays = getattr(args, 'tune_encoder_weight_decays', None)
    head_lrs = _parse_positive_float_list(
        tune_head_lrs or args.tune_lrs,
        name='--tune-head-lrs' if tune_head_lrs else '--tune-lrs',
    )
    encoder_lrs = _parse_positive_float_list(
        tune_encoder_lrs or args.tune_lrs,
        name='--tune-encoder-lrs' if tune_encoder_lrs else '--tune-lrs',
    )
    dropouts = _parse_dropout_list(args.tune_dropouts)
    head_weight_decays = _parse_nonnegative_float_list(
        tune_head_weight_decays or '0',
        name='--tune-head-weight-decays',
    )
    encoder_weight_decays = _parse_nonnegative_float_list(
        tune_encoder_weight_decays or args.tune_weight_decays,
        name=(
            '--tune-encoder-weight-decays'
            if tune_encoder_weight_decays
            else '--tune-weight-decays'
        ),
    )
    swa_start_epochs = [
        epoch
        for epoch in _parse_positive_int_list(
            args.tune_swa_start_epochs,
            name='--tune-swa-start-epochs',
        )
        if epoch <= tune_epochs
    ]
    swa_lrs = _parse_positive_float_list(args.tune_swa_lrs, name='--tune-swa-lrs')
    anneal_epochs = max(1, int(args.swa_anneal_epochs))

    base_products = itertools.product(
        head_lrs,
        encoder_lrs,
        dropouts,
        head_weight_decays,
        encoder_weight_decays,
    )
    candidates = [
        _ValidationTuneCandidate(
            head_lr=head_lr,
            encoder_lr=encoder_lr,
            dropout=dropout,
            head_weight_decay=head_weight_decay,
            encoder_weight_decay=encoder_weight_decay,
            swa=False,
            swa_start_epoch=None,
            swa_lr=None,
            swa_anneal_epochs=anneal_epochs,
        )
        for (
            head_lr,
            encoder_lr,
            dropout,
            head_weight_decay,
            encoder_weight_decay,
        ) in base_products
    ]
    candidates.extend(
        _ValidationTuneCandidate(
            head_lr=head_lr,
            encoder_lr=encoder_lr,
            dropout=dropout,
            head_weight_decay=head_weight_decay,
            encoder_weight_decay=encoder_weight_decay,
            swa=True,
            swa_start_epoch=swa_start_epoch,
            swa_lr=swa_lr,
            swa_anneal_epochs=anneal_epochs,
        )
        for (
            head_lr,
            encoder_lr,
            dropout,
            head_weight_decay,
            encoder_weight_decay,
            swa_start_epoch,
            swa_lr,
        ) in itertools.product(
            head_lrs,
            encoder_lrs,
            dropouts,
            head_weight_decays,
            encoder_weight_decays,
            swa_start_epochs,
            swa_lrs,
        )
        if swa_lr <= min(head_lr, encoder_lr)
    )
    candidates = _deduplicate_candidates(candidates)
    rng = random.Random(args.tune_seed if args.tune_seed is not None else args.seed)
    rng.shuffle(candidates)
    return candidates[: int(args.tune_trials)]


def _deduplicate_candidates(
    candidates: list[_ValidationTuneCandidate],
) -> list[_ValidationTuneCandidate]:
    """
    Drop duplicate tuning candidates while preserving order.
    """
    seen: set[_ValidationTuneCandidate] = set()
    out: list[_ValidationTuneCandidate] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out


def _parse_positive_float_list(value: str, *, name: str) -> list[float]:
    """
    Parse a comma-separated list of positive floats.
    """
    values = _parse_float_list(value, name=name)
    if any(item <= 0.0 for item in values):
        raise SystemExit(f'{name} must contain only positive values.')
    return values


def _parse_nonnegative_float_list(value: str, *, name: str) -> list[float]:
    """
    Parse a comma-separated list of non-negative floats.
    """
    values = _parse_float_list(value, name=name)
    if any(item < 0.0 for item in values):
        raise SystemExit(f'{name} must contain only non-negative values.')
    return values


def _parse_dropout_list(value: str) -> list[float]:
    """
    Parse a comma-separated list of dropout probabilities.
    """
    values = _parse_float_list(value, name='--tune-dropouts')
    if any(item < 0.0 or item >= 1.0 for item in values):
        raise SystemExit('--tune-dropouts values must be in [0, 1).')
    return values


def _parse_float_list(value: str, *, name: str) -> list[float]:
    """
    Parse a comma-separated float list and reject empty lists.
    """
    try:
        values = [float(part.strip()) for part in value.split(',') if part.strip()]
    except ValueError as exc:
        raise SystemExit(f'{name} must be a comma-separated float list.') from exc
    if not values:
        raise SystemExit(f'{name} must contain at least one value.')
    return values


def _parse_positive_int_list(value: str, *, name: str) -> list[int]:
    """
    Parse a comma-separated list of positive integers.
    """
    try:
        values = [int(part.strip()) for part in value.split(',') if part.strip()]
    except ValueError as exc:
        raise SystemExit(f'{name} must be a comma-separated integer list.') from exc
    if not values:
        raise SystemExit(f'{name} must contain at least one value.')
    if any(item < 1 for item in values):
        raise SystemExit(f'{name} must contain only positive values.')
    return values


def _config_bond_break_geometry_features(config: MiraFragConfig) -> bool:
    """Return the effective saved bond-event geometry setting."""
    return bool(
        getattr(config, 'bond_break_geometry_features', False)
        or getattr(config, 'fragment_action_geometry_features', False)
    )


def _requested_bond_break_geometry_features(
    args: argparse.Namespace,
) -> bool | None:
    """Return the explicit CLI geometry request, including the legacy alias."""
    value = getattr(args, 'bond_break_geometry_features', None)
    if value is not None:
        return bool(value)
    value = getattr(args, 'fragment_action_geometry_features', None)
    if value is not None:
        return bool(value)
    return None


def _new_model_bond_break_geometry_features(
    args: argparse.Namespace,
    *,
    action_primary_layers: int,
    action_bond_gnn_layers: int,
) -> bool:
    """Default direct geometry on for new prediction-affecting bond branches."""
    requested = _requested_bond_break_geometry_features(args)
    if requested is not None:
        return requested
    return int(action_primary_layers) > 0 or int(action_bond_gnn_layers) > 0


def _checkpoint_bond_break_geometry_features(
    args: argparse.Namespace,
    config: MiraFragConfig,
    *,
    target_action_primary_layers: int,
    target_action_bond_gnn_layers: int,
) -> bool:
    """Resolve geometry defaults without changing old action-branch checkpoints."""
    requested = _requested_bond_break_geometry_features(args)
    if requested is not None:
        return requested
    current = _config_bond_break_geometry_features(config)
    current_action_branch = (
        int(getattr(config, 'fragment_action_primary_layers', 0)) > 0
        or int(getattr(config, 'fragment_action_bond_gnn_layers', 0)) > 0
    )
    target_action_branch = (
        int(target_action_primary_layers) > 0 or int(target_action_bond_gnn_layers) > 0
    )
    if target_action_branch and not current_action_branch:
        return True
    return current


def _maybe_rebuild_fragment_bond_break_model(
    model: MiraFragModel,
    args: argparse.Namespace,
    *,
    device: str | torch.device,
) -> MiraFragModel:
    """
    Rebuild a checkpoint model when opt-in fragmentation head modes change.

    Checkpoints are strict by default. This helper keeps that behavior for all
    existing parameters and only tolerates missing or unexpected keys belonging
    to optional path or bond-break auxiliary branches.
    """
    requested_bond_layers = getattr(args, 'fragment_bond_break_layers', None)
    requested_path_primary = getattr(args, 'fragment_path_primary', None)
    requested_path_layers = getattr(args, 'fragment_path_layers', None)
    requested_action_layers = getattr(args, 'fragment_action_path_layers', None)
    requested_action_primary_layers = getattr(
        args, 'fragment_action_primary_layers', None
    )
    requested_action_primary_ce_gate = getattr(
        args, 'fragment_action_primary_ce_gate', None
    )
    requested_action_bond_gnn_layers = getattr(
        args, 'fragment_action_bond_gnn_layers', None
    )
    requested_fragnnet_dag_layers = getattr(args, 'fragnnet_dag_layers', None)
    requested_fragnnet_dag_num_hs = getattr(args, 'fragnnet_dag_num_hs', None)
    requested_ce_basis = getattr(args, 'ce_basis_features', None)
    requested_ce_embedding = getattr(args, 'ce_embedding', None)
    requested_ce_fourier_frequencies = getattr(args, 'ce_fourier_frequencies', None)
    requested_aimnet_charge = getattr(args, 'aimnet_charge_features', None)
    requested_aimnet_multipass = getattr(args, 'aimnet_multipass_features', None)
    requested_candidate_transformer_layers = getattr(
        args, 'candidate_transformer_layers', None
    )
    requested_candidate_transformer_heads = getattr(
        args, 'candidate_transformer_heads', None
    )
    requested_candidate_transformer_max_tokens = getattr(
        args, 'candidate_transformer_max_tokens', None
    )
    requested_candidate_suppression_gate_hidden_dim = getattr(
        args, 'candidate_suppression_gate_hidden_dim', None
    )
    requested_candidate_suppression_gate_dropout = getattr(
        args, 'candidate_suppression_gate_dropout', None
    )
    requested_candidate_suppression_gate_initial_penalty = getattr(
        args, 'candidate_suppression_gate_initial_penalty', None
    )
    requested_conditional_expert_heads = getattr(args, 'conditional_expert_heads', None)
    requested_conditional_expert_hidden_dim = getattr(
        args, 'conditional_expert_hidden_dim', None
    )
    requested_conditional_expert_dropout = getattr(
        args, 'conditional_expert_dropout', None
    )
    requested_molecule_descriptor_features = getattr(
        args, 'molecule_descriptor_features', None
    )
    requested_molecule_descriptor_hidden_dim = getattr(
        args, 'molecule_descriptor_hidden_dim', None
    )
    requested_molecule_descriptor_dropout = getattr(
        args, 'molecule_descriptor_dropout', None
    )
    requested_physical_bond_features = getattr(args, 'physical_bond_features', None)
    requested_physical_bond_feature_columns = (
        parse_physical_bond_feature_columns(args.physical_bond_feature_columns)
        if getattr(args, 'physical_bond_feature_columns', None) is not None
        else None
    )

    requested_bond_layers = (
        None if requested_bond_layers is None else int(requested_bond_layers)
    )
    if requested_bond_layers is not None and requested_bond_layers < 0:
        raise SystemExit('--fragment-bond-break-layers must be nonnegative.')
    requested_path_layers = (
        None if requested_path_layers is None else int(requested_path_layers)
    )
    if requested_path_layers is not None and requested_path_layers < 0:
        raise SystemExit('--fragment-path-layers must be nonnegative.')
    requested_action_layers = (
        None if requested_action_layers is None else int(requested_action_layers)
    )
    if requested_action_layers is not None and requested_action_layers < 0:
        raise SystemExit('--fragment-action-path-layers must be nonnegative.')
    requested_action_primary_layers = (
        None
        if requested_action_primary_layers is None
        else int(requested_action_primary_layers)
    )
    if (
        requested_action_primary_layers is not None
        and requested_action_primary_layers < 0
    ):
        raise SystemExit('--fragment-action-primary-layers must be nonnegative.')
    requested_action_bond_gnn_layers = (
        None
        if requested_action_bond_gnn_layers is None
        else int(requested_action_bond_gnn_layers)
    )
    if (
        requested_action_bond_gnn_layers is not None
        and requested_action_bond_gnn_layers < 0
    ):
        raise SystemExit('--fragment-action-bond-gnn-layers must be nonnegative.')
    requested_fragnnet_dag_layers = (
        None
        if requested_fragnnet_dag_layers is None
        else int(requested_fragnnet_dag_layers)
    )
    if requested_fragnnet_dag_layers is not None and requested_fragnnet_dag_layers < 0:
        raise SystemExit('--fragnnet-dag-layers must be nonnegative.')
    requested_fragnnet_dag_num_hs = (
        None
        if requested_fragnnet_dag_num_hs is None
        else int(requested_fragnnet_dag_num_hs)
    )
    if requested_fragnnet_dag_num_hs is not None and requested_fragnnet_dag_num_hs < 0:
        raise SystemExit('--fragnnet-dag-num-hs must be nonnegative.')
    requested_ce_fourier_frequencies = (
        None
        if requested_ce_fourier_frequencies is None
        else int(requested_ce_fourier_frequencies)
    )
    if (
        requested_ce_fourier_frequencies is not None
        and requested_ce_fourier_frequencies <= 0
    ):
        raise SystemExit('--ce-fourier-frequencies must be positive.')
    requested_candidate_transformer_layers = (
        None
        if requested_candidate_transformer_layers is None
        else int(requested_candidate_transformer_layers)
    )
    if (
        requested_candidate_transformer_layers is not None
        and requested_candidate_transformer_layers < 0
    ):
        raise SystemExit('--candidate-transformer-layers must be nonnegative.')
    requested_candidate_transformer_heads = (
        None
        if requested_candidate_transformer_heads is None
        else int(requested_candidate_transformer_heads)
    )
    if (
        requested_candidate_transformer_heads is not None
        and requested_candidate_transformer_heads <= 0
    ):
        raise SystemExit('--candidate-transformer-heads must be positive.')
    requested_candidate_transformer_max_tokens = (
        None
        if requested_candidate_transformer_max_tokens is None
        else int(requested_candidate_transformer_max_tokens)
    )
    if (
        requested_candidate_transformer_max_tokens is not None
        and requested_candidate_transformer_max_tokens <= 0
    ):
        raise SystemExit('--candidate-transformer-max-tokens must be positive.')
    requested_candidate_suppression_gate_hidden_dim = (
        None
        if requested_candidate_suppression_gate_hidden_dim is None
        else int(requested_candidate_suppression_gate_hidden_dim)
    )
    if (
        requested_candidate_suppression_gate_hidden_dim is not None
        and requested_candidate_suppression_gate_hidden_dim < 0
    ):
        raise SystemExit('--candidate-suppression-gate-hidden-dim must be nonnegative.')
    requested_candidate_suppression_gate_dropout = (
        None
        if requested_candidate_suppression_gate_dropout is None
        else float(requested_candidate_suppression_gate_dropout)
    )
    if (
        requested_candidate_suppression_gate_dropout is not None
        and requested_candidate_suppression_gate_dropout < 0
    ):
        raise SystemExit('--candidate-suppression-gate-dropout must be nonnegative.')
    requested_candidate_suppression_gate_initial_penalty = (
        None
        if requested_candidate_suppression_gate_initial_penalty is None
        else float(requested_candidate_suppression_gate_initial_penalty)
    )
    if (
        requested_candidate_suppression_gate_initial_penalty is not None
        and requested_candidate_suppression_gate_initial_penalty <= 0
    ):
        raise SystemExit(
            '--candidate-suppression-gate-initial-penalty must be positive.'
        )
    requested_conditional_expert_heads = (
        None
        if requested_conditional_expert_heads is None
        else int(requested_conditional_expert_heads)
    )
    if (
        requested_conditional_expert_heads is not None
        and requested_conditional_expert_heads <= 0
    ):
        raise SystemExit('--conditional-expert-heads must be positive.')
    requested_conditional_expert_hidden_dim = (
        None
        if requested_conditional_expert_hidden_dim is None
        else int(requested_conditional_expert_hidden_dim)
    )
    if (
        requested_conditional_expert_hidden_dim is not None
        and requested_conditional_expert_hidden_dim <= 0
    ):
        raise SystemExit('--conditional-expert-hidden-dim must be positive.')
    requested_conditional_expert_dropout = (
        None
        if requested_conditional_expert_dropout is None
        else float(requested_conditional_expert_dropout)
    )
    if (
        requested_conditional_expert_dropout is not None
        and requested_conditional_expert_dropout < 0
    ):
        raise SystemExit('--conditional-expert-dropout must be nonnegative.')
    requested_molecule_descriptor_hidden_dim = (
        None
        if requested_molecule_descriptor_hidden_dim is None
        else int(requested_molecule_descriptor_hidden_dim)
    )
    if (
        requested_molecule_descriptor_hidden_dim is not None
        and requested_molecule_descriptor_hidden_dim <= 0
    ):
        raise SystemExit('--molecule-descriptor-hidden-dim must be positive.')
    requested_molecule_descriptor_dropout = (
        None
        if requested_molecule_descriptor_dropout is None
        else float(requested_molecule_descriptor_dropout)
    )
    if (
        requested_molecule_descriptor_dropout is not None
        and requested_molecule_descriptor_dropout < 0
    ):
        raise SystemExit('--molecule-descriptor-dropout must be nonnegative.')

    target_bond_layers = (
        int(getattr(model.config, 'fragment_bond_break_layers', 0))
        if requested_bond_layers is None
        else requested_bond_layers
    )
    target_path_layers = (
        int(getattr(model.config, 'fragment_path_layers', 0))
        if requested_path_layers is None
        else requested_path_layers
    )
    target_path_primary = (
        bool(getattr(model.config, 'fragment_path_primary', False))
        if requested_path_primary is None
        else bool(requested_path_primary)
    )
    target_action_layers = (
        int(getattr(model.config, 'fragment_action_path_layers', 0))
        if requested_action_layers is None
        else requested_action_layers
    )
    target_action_primary_layers = (
        int(getattr(model.config, 'fragment_action_primary_layers', 0))
        if requested_action_primary_layers is None
        else requested_action_primary_layers
    )
    target_action_primary_ce_gate = (
        bool(getattr(model.config, 'fragment_action_primary_ce_gate', False))
        if requested_action_primary_ce_gate is None
        else bool(requested_action_primary_ce_gate)
    )
    target_action_bond_gnn_layers = (
        int(getattr(model.config, 'fragment_action_bond_gnn_layers', 0))
        if requested_action_bond_gnn_layers is None
        else int(requested_action_bond_gnn_layers)
    )
    target_bond_break_geometry_features = _checkpoint_bond_break_geometry_features(
        args,
        model.config,
        target_action_primary_layers=target_action_primary_layers,
        target_action_bond_gnn_layers=target_action_bond_gnn_layers,
    )
    target_fragnnet_dag_layers = (
        int(getattr(model.config, 'fragnnet_dag_layers', 0))
        if requested_fragnnet_dag_layers is None
        else requested_fragnnet_dag_layers
    )
    target_fragnnet_dag_num_hs = (
        int(getattr(model.config, 'fragnnet_dag_num_hs', 4))
        if requested_fragnnet_dag_num_hs is None
        else requested_fragnnet_dag_num_hs
    )
    target_ce_basis = (
        bool(getattr(model.config, 'ce_basis_features', False))
        if requested_ce_basis is None
        else bool(requested_ce_basis)
    )
    current_config_ce_embedding = str(
        getattr(model.config, 'ce_embedding', 'scalar') or 'scalar'
    ).lower()
    if target_ce_basis and current_config_ce_embedding == 'scalar':
        current_config_ce_embedding = 'basis'
    target_ce_embedding = (
        current_config_ce_embedding
        if requested_ce_embedding is None
        else str(requested_ce_embedding).lower()
    )
    if requested_ce_embedding is None and requested_ce_basis is not None:
        target_ce_embedding = 'basis' if target_ce_basis else 'scalar'
    if target_ce_embedding not in {'scalar', 'basis', 'fourier'}:
        raise SystemExit('--ce-embedding must be one of: scalar, basis, fourier.')
    target_ce_basis = target_ce_embedding == 'basis'
    target_ce_fourier_frequencies = (
        int(getattr(model.config, 'ce_fourier_frequencies', 8))
        if requested_ce_fourier_frequencies is None
        else requested_ce_fourier_frequencies
    )
    target_aimnet_charge = (
        bool(getattr(model.config, 'aimnet_charge_features', False))
        if requested_aimnet_charge is None
        else bool(requested_aimnet_charge)
    )
    target_aimnet_multipass = (
        bool(getattr(model.config, 'aimnet_multipass_features', False))
        if requested_aimnet_multipass is None
        else bool(requested_aimnet_multipass)
    )
    target_candidate_transformer_layers = (
        int(getattr(model.config, 'candidate_transformer_layers', 0))
        if requested_candidate_transformer_layers is None
        else requested_candidate_transformer_layers
    )
    target_candidate_transformer_heads = (
        int(getattr(model.config, 'candidate_transformer_heads', 8))
        if requested_candidate_transformer_heads is None
        else requested_candidate_transformer_heads
    )
    target_candidate_transformer_max_tokens = (
        int(getattr(model.config, 'candidate_transformer_max_tokens', 256))
        if requested_candidate_transformer_max_tokens is None
        else requested_candidate_transformer_max_tokens
    )
    target_candidate_suppression_gate_hidden_dim = (
        int(getattr(model.config, 'candidate_suppression_gate_hidden_dim', 0))
        if requested_candidate_suppression_gate_hidden_dim is None
        else requested_candidate_suppression_gate_hidden_dim
    )
    target_candidate_suppression_gate_dropout = (
        float(getattr(model.config, 'candidate_suppression_gate_dropout', 0.0))
        if requested_candidate_suppression_gate_dropout is None
        else requested_candidate_suppression_gate_dropout
    )
    target_candidate_suppression_gate_initial_penalty = (
        float(getattr(model.config, 'candidate_suppression_gate_initial_penalty', 1e-6))
        if requested_candidate_suppression_gate_initial_penalty is None
        else requested_candidate_suppression_gate_initial_penalty
    )
    target_conditional_expert_heads = (
        int(getattr(model.config, 'conditional_expert_heads', 1))
        if requested_conditional_expert_heads is None
        else requested_conditional_expert_heads
    )
    target_conditional_expert_hidden_dim = (
        int(getattr(model.config, 'conditional_expert_hidden_dim', 128))
        if requested_conditional_expert_hidden_dim is None
        else requested_conditional_expert_hidden_dim
    )
    target_conditional_expert_dropout = (
        float(getattr(model.config, 'conditional_expert_dropout', 0.0))
        if requested_conditional_expert_dropout is None
        else requested_conditional_expert_dropout
    )
    target_molecule_descriptor_features = (
        bool(getattr(model.config, 'molecule_descriptor_features', False))
        if requested_molecule_descriptor_features is None
        else bool(requested_molecule_descriptor_features)
    )
    target_molecule_descriptor_hidden_dim = (
        int(getattr(model.config, 'molecule_descriptor_hidden_dim', 128))
        if requested_molecule_descriptor_hidden_dim is None
        else requested_molecule_descriptor_hidden_dim
    )
    target_molecule_descriptor_dropout = (
        float(getattr(model.config, 'molecule_descriptor_dropout', 0.0))
        if requested_molecule_descriptor_dropout is None
        else requested_molecule_descriptor_dropout
    )
    target_physical_bond_features = (
        bool(getattr(model.config, 'physical_bond_features', False))
        if requested_physical_bond_features is None
        else bool(requested_physical_bond_features)
    )
    target_physical_bond_feature_columns = (
        tuple(getattr(model.config, 'physical_bond_feature_columns', ()) or ())
        if requested_physical_bond_feature_columns is None
        else tuple(requested_physical_bond_feature_columns)
    )
    if not target_physical_bond_features:
        target_physical_bond_feature_columns = ()

    if target_path_primary and target_path_layers <= 0:
        raise SystemExit('--fragment-path-primary requires --fragment-path-layers > 0.')

    current_bond_layers = int(getattr(model.config, 'fragment_bond_break_layers', 0))
    current_path_layers = int(getattr(model.config, 'fragment_path_layers', 0))
    current_path_primary = bool(getattr(model.config, 'fragment_path_primary', False))
    current_action_layers = int(getattr(model.config, 'fragment_action_path_layers', 0))
    current_action_primary_layers = int(
        getattr(model.config, 'fragment_action_primary_layers', 0)
    )
    current_action_primary_ce_gate = bool(
        getattr(model.config, 'fragment_action_primary_ce_gate', False)
    )
    current_bond_break_geometry_features = _config_bond_break_geometry_features(
        model.config
    )
    current_action_bond_gnn_layers = int(
        getattr(model.config, 'fragment_action_bond_gnn_layers', 0)
    )
    current_fragnnet_dag_layers = int(getattr(model.config, 'fragnnet_dag_layers', 0))
    current_fragnnet_dag_num_hs = int(getattr(model.config, 'fragnnet_dag_num_hs', 4))
    current_ce_basis = bool(getattr(model.config, 'ce_basis_features', False))
    current_ce_embedding = current_config_ce_embedding
    current_ce_fourier_frequencies = int(
        getattr(model.config, 'ce_fourier_frequencies', 8)
    )
    current_aimnet_charge = bool(getattr(model.config, 'aimnet_charge_features', False))
    current_aimnet_multipass = bool(
        getattr(model.config, 'aimnet_multipass_features', False)
    )
    current_candidate_transformer_layers = int(
        getattr(model.config, 'candidate_transformer_layers', 0)
    )
    current_candidate_transformer_heads = int(
        getattr(model.config, 'candidate_transformer_heads', 8)
    )
    current_candidate_transformer_max_tokens = int(
        getattr(model.config, 'candidate_transformer_max_tokens', 256)
    )
    current_candidate_suppression_gate_hidden_dim = int(
        getattr(model.config, 'candidate_suppression_gate_hidden_dim', 0)
    )
    current_candidate_suppression_gate_dropout = float(
        getattr(model.config, 'candidate_suppression_gate_dropout', 0.0)
    )
    current_candidate_suppression_gate_initial_penalty = float(
        getattr(model.config, 'candidate_suppression_gate_initial_penalty', 1e-6)
    )
    current_conditional_expert_heads = int(
        getattr(model.config, 'conditional_expert_heads', 1)
    )
    current_conditional_expert_hidden_dim = int(
        getattr(model.config, 'conditional_expert_hidden_dim', 128)
    )
    current_conditional_expert_dropout = float(
        getattr(model.config, 'conditional_expert_dropout', 0.0)
    )
    current_molecule_descriptor_features = bool(
        getattr(model.config, 'molecule_descriptor_features', False)
    )
    current_molecule_descriptor_hidden_dim = int(
        getattr(model.config, 'molecule_descriptor_hidden_dim', 128)
    )
    current_molecule_descriptor_dropout = float(
        getattr(model.config, 'molecule_descriptor_dropout', 0.0)
    )
    current_physical_bond_features = bool(
        getattr(model.config, 'physical_bond_features', False)
    )
    current_physical_bond_feature_columns = tuple(
        getattr(model.config, 'physical_bond_feature_columns', ()) or ()
    )
    physical_bond_feature_schema_changed = (
        target_physical_bond_features != current_physical_bond_features
        or target_physical_bond_feature_columns != current_physical_bond_feature_columns
    )
    action_event_feature_schema_changed = (
        physical_bond_feature_schema_changed
        or target_bond_break_geometry_features != current_bond_break_geometry_features
    )
    if (
        target_bond_layers == current_bond_layers
        and target_path_layers == current_path_layers
        and target_path_primary == current_path_primary
        and target_action_layers == current_action_layers
        and target_action_primary_layers == current_action_primary_layers
        and target_action_primary_ce_gate == current_action_primary_ce_gate
        and target_bond_break_geometry_features == current_bond_break_geometry_features
        and target_action_bond_gnn_layers == current_action_bond_gnn_layers
        and target_fragnnet_dag_layers == current_fragnnet_dag_layers
        and target_fragnnet_dag_num_hs == current_fragnnet_dag_num_hs
        and target_ce_basis == current_ce_basis
        and target_ce_embedding == current_ce_embedding
        and target_ce_fourier_frequencies == current_ce_fourier_frequencies
        and target_aimnet_charge == current_aimnet_charge
        and target_aimnet_multipass == current_aimnet_multipass
        and target_candidate_transformer_layers == current_candidate_transformer_layers
        and target_candidate_transformer_heads == current_candidate_transformer_heads
        and target_candidate_transformer_max_tokens
        == current_candidate_transformer_max_tokens
        and target_candidate_suppression_gate_hidden_dim
        == current_candidate_suppression_gate_hidden_dim
        and target_candidate_suppression_gate_dropout
        == current_candidate_suppression_gate_dropout
        and target_candidate_suppression_gate_initial_penalty
        == current_candidate_suppression_gate_initial_penalty
        and target_conditional_expert_heads == current_conditional_expert_heads
        and target_conditional_expert_hidden_dim
        == current_conditional_expert_hidden_dim
        and target_conditional_expert_dropout == current_conditional_expert_dropout
        and target_molecule_descriptor_features == current_molecule_descriptor_features
        and target_molecule_descriptor_hidden_dim
        == current_molecule_descriptor_hidden_dim
        and target_molecule_descriptor_dropout == current_molecule_descriptor_dropout
        and not action_event_feature_schema_changed
    ):
        return model

    state_dict = model.state_dict()
    config = replace(
        model.config,
        fragment_bond_break_layers=target_bond_layers,
        physical_bond_features=target_physical_bond_features,
        physical_bond_feature_columns=target_physical_bond_feature_columns,
        fragment_path_layers=target_path_layers,
        fragment_path_primary=target_path_primary,
        fragment_action_path_layers=target_action_layers,
        fragment_action_primary_layers=target_action_primary_layers,
        fragment_action_primary_ce_gate=target_action_primary_ce_gate,
        bond_break_geometry_features=target_bond_break_geometry_features,
        fragment_action_geometry_features=target_bond_break_geometry_features,
        fragment_action_bond_gnn_layers=target_action_bond_gnn_layers,
        fragnnet_dag_layers=target_fragnnet_dag_layers,
        fragnnet_dag_num_hs=target_fragnnet_dag_num_hs,
        ce_basis_features=target_ce_basis,
        ce_embedding=target_ce_embedding,
        ce_fourier_frequencies=target_ce_fourier_frequencies,
        aimnet_charge_features=target_aimnet_charge,
        aimnet_multipass_features=target_aimnet_multipass,
        candidate_transformer_layers=target_candidate_transformer_layers,
        candidate_transformer_heads=target_candidate_transformer_heads,
        candidate_transformer_max_tokens=target_candidate_transformer_max_tokens,
        candidate_suppression_gate_hidden_dim=(
            target_candidate_suppression_gate_hidden_dim
        ),
        candidate_suppression_gate_dropout=target_candidate_suppression_gate_dropout,
        candidate_suppression_gate_initial_penalty=(
            target_candidate_suppression_gate_initial_penalty
        ),
        conditional_expert_heads=target_conditional_expert_heads,
        conditional_expert_hidden_dim=target_conditional_expert_hidden_dim,
        conditional_expert_dropout=target_conditional_expert_dropout,
        molecule_descriptor_features=target_molecule_descriptor_features,
        molecule_descriptor_hidden_dim=target_molecule_descriptor_hidden_dim,
        molecule_descriptor_dropout=target_molecule_descriptor_dropout,
    )
    rebuilt = MiraFragModel(
        model.encoder,
        metadata_config=model.metadata_config,
        config=config,
    ).to(device)
    filtered_state_dict = dict(state_dict)
    if action_event_feature_schema_changed:
        for key in list(filtered_state_dict):
            if key.startswith(
                (
                    'head.bond_break_pair_scorer.0.',
                    'head.fragment_action_primary_pair_scorer.0.',
                    'head.fragment_action_bond_gnn_encoder.0.',
                )
            ):
                filtered_state_dict.pop(key, None)
    collision_weight_key = 'head.collision_encoder.0.weight'
    old_collision_weight = filtered_state_dict.get(collision_weight_key)
    new_collision_weight = rebuilt.state_dict().get(collision_weight_key)
    copy_collision_weight = (
        old_collision_weight is not None
        and new_collision_weight is not None
        and tuple(old_collision_weight.shape) != tuple(new_collision_weight.shape)
        and old_collision_weight.ndim == 2
        and new_collision_weight.ndim == 2
        and old_collision_weight.shape[0] == new_collision_weight.shape[0]
    )
    if copy_collision_weight:
        filtered_state_dict.pop(collision_weight_key, None)
    incompatible = rebuilt.load_state_dict(filtered_state_dict, strict=False)
    if copy_collision_weight:
        with torch.no_grad():
            target_weight = rebuilt.head.collision_encoder[0].weight
            target_weight.zero_()
            width = min(old_collision_weight.shape[1], target_weight.shape[1])
            target_weight[:, :width].copy_(old_collision_weight[:, :width])
    allowed_prefixes = (
        'head.bond_break_',
        'head.fragment_path_',
        'head.fragment_action_',
        'head.fragnnet_dag_',
        'head.aimnet_charge_',
        'head.formula_count_',
        'head.candidate_transformer_',
        'head.candidate_transformer.',
        'head.candidate_suppression_gate',
        'head.conditional_expert_',
        'head.molecule_descriptor_',
        'aimnet_multipass_adapter.',
        'head.collision_encoder.0.weight',
    )
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith(allowed_prefixes)
    ]
    missing = [
        key for key in incompatible.missing_keys if not key.startswith(allowed_prefixes)
    ]
    if unexpected or missing:
        raise RuntimeError(
            'Unexpected checkpoint mismatch while enabling fragmentation head mode: '
            f'missing={missing} unexpected={unexpected}'
        )
    branch_mismatch = len(incompatible.missing_keys) + len(incompatible.unexpected_keys)
    if branch_mismatch:
        print(
            'Loaded checkpoint with '
            f'{branch_mismatch} fragmentation head parameters initialized from defaults.'
        )
    return rebuilt


def _maybe_rebuild_encoder_bond_adapter_model(
    model: MiraFragModel,
    args: argparse.Namespace,
    *,
    device: str | torch.device,
) -> MiraFragModel:
    """
    Rebuild checkpoint model when the optional post-encoder bond adapter changes.
    """
    requested_layers = getattr(args, 'encoder_bond_adapter_layers', None)
    requested_dim = getattr(args, 'encoder_bond_adapter_feature_dim', None)
    requested_dropout = getattr(args, 'encoder_bond_adapter_dropout', None)
    target_layers = (
        int(getattr(model.config, 'encoder_bond_adapter_layers', 0))
        if requested_layers is None
        else int(requested_layers)
    )
    target_dim = (
        int(getattr(model.config, 'encoder_bond_adapter_feature_dim', 128))
        if requested_dim is None
        else int(requested_dim)
    )
    target_dropout = (
        float(getattr(model.config, 'encoder_bond_adapter_dropout', 0.0))
        if requested_dropout is None
        else float(requested_dropout)
    )
    if target_layers < 0:
        raise SystemExit('--encoder-bond-adapter-layers must be nonnegative.')
    if target_dim <= 0:
        raise SystemExit('--encoder-bond-adapter-feature-dim must be positive.')
    if target_dropout < 0:
        raise SystemExit('--encoder-bond-adapter-dropout must be nonnegative.')
    if (
        target_layers == int(getattr(model.config, 'encoder_bond_adapter_layers', 0))
        and target_dim
        == int(getattr(model.config, 'encoder_bond_adapter_feature_dim', 128))
        and target_dropout
        == float(getattr(model.config, 'encoder_bond_adapter_dropout', 0.0))
    ):
        return model
    state_dict = model.state_dict()
    config = replace(
        model.config,
        encoder_bond_adapter_layers=target_layers,
        encoder_bond_adapter_feature_dim=target_dim,
        encoder_bond_adapter_dropout=target_dropout,
    )
    rebuilt = MiraFragModel(
        model.encoder,
        metadata_config=model.metadata_config,
        config=config,
    ).to(device)
    incompatible = rebuilt.load_state_dict(state_dict, strict=False)
    allowed_prefix = 'encoder_bond_adapter.'
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith(allowed_prefix)
    ]
    missing = [
        key for key in incompatible.missing_keys if not key.startswith(allowed_prefix)
    ]
    if unexpected or missing:
        raise RuntimeError(
            'Unexpected checkpoint mismatch while changing encoder bond adapter: '
            f'missing={missing} unexpected={unexpected}'
        )
    branch_mismatch = len(incompatible.missing_keys) + len(incompatible.unexpected_keys)
    if branch_mismatch:
        print(
            'Loaded checkpoint with '
            f'{branch_mismatch} encoder bond-adapter parameters initialized from defaults.'
        )
    return rebuilt


def _maybe_rebuild_encoder_metadata_adapter_model(
    model: MiraFragModel,
    args: argparse.Namespace,
    *,
    device: str | torch.device,
) -> MiraFragModel:
    """
    Rebuild checkpoint model when the optional metadata-conditioned adapter changes.
    """
    requested_layers = getattr(args, 'encoder_metadata_adapter_layers', None)
    requested_dim = getattr(args, 'encoder_metadata_adapter_feature_dim', None)
    requested_dropout = getattr(args, 'encoder_metadata_adapter_dropout', None)
    target_layers = (
        int(getattr(model.config, 'encoder_metadata_adapter_layers', 0))
        if requested_layers is None
        else int(requested_layers)
    )
    target_dim = (
        int(getattr(model.config, 'encoder_metadata_adapter_feature_dim', 128))
        if requested_dim is None
        else int(requested_dim)
    )
    target_dropout = (
        float(getattr(model.config, 'encoder_metadata_adapter_dropout', 0.0))
        if requested_dropout is None
        else float(requested_dropout)
    )
    if target_layers < 0:
        raise SystemExit('--encoder-metadata-adapter-layers must be nonnegative.')
    if target_dim <= 0:
        raise SystemExit('--encoder-metadata-adapter-feature-dim must be positive.')
    if target_dropout < 0:
        raise SystemExit('--encoder-metadata-adapter-dropout must be nonnegative.')
    if (
        target_layers
        == int(getattr(model.config, 'encoder_metadata_adapter_layers', 0))
        and target_dim
        == int(getattr(model.config, 'encoder_metadata_adapter_feature_dim', 128))
        and target_dropout
        == float(getattr(model.config, 'encoder_metadata_adapter_dropout', 0.0))
    ):
        return model
    state_dict = model.state_dict()
    config = replace(
        model.config,
        encoder_metadata_adapter_layers=target_layers,
        encoder_metadata_adapter_feature_dim=target_dim,
        encoder_metadata_adapter_dropout=target_dropout,
    )
    rebuilt = MiraFragModel(
        model.encoder,
        metadata_config=model.metadata_config,
        config=config,
    ).to(device)
    incompatible = rebuilt.load_state_dict(state_dict, strict=False)
    allowed_prefix = 'encoder_metadata_adapter.'
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith(allowed_prefix)
    ]
    missing = [
        key for key in incompatible.missing_keys if not key.startswith(allowed_prefix)
    ]
    if unexpected or missing:
        raise RuntimeError(
            'Unexpected checkpoint mismatch while changing encoder metadata adapter: '
            f'missing={missing} unexpected={unexpected}'
        )
    branch_mismatch = len(incompatible.missing_keys) + len(incompatible.unexpected_keys)
    if branch_mismatch:
        print(
            'Loaded checkpoint with '
            f'{branch_mismatch} encoder metadata-adapter parameters initialized from defaults.'
        )
    return rebuilt


def _set_head_dropout(model: MiraFragModel, dropout: float) -> None:
    """
    Change only the spectrum-head dropout probability for a tuning trial.
    """
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError('dropout must be in [0, 1).')
    model.config.dropout = float(dropout)
    for module in model.head.modules():
        if isinstance(module, nn.Dropout):
            module.p = float(dropout)


def _tune_score_from_val_stats(stats: dict[str, float], metric: str) -> float:
    """
    Extract the selected validation-tuning metric from one validation result.
    """
    if metric == 'val_cosine':
        return float(stats['cosine'])
    if metric == 'val_loss':
        return float(stats['loss'])
    raise ValueError(f'Unsupported tuning metric: {metric}')


def _best_tune_score(history: dict[str, list[float]], metric: str) -> float:
    """
    Return the best live-or-SWA validation score from a trial history.
    """
    values = [
        value
        for key in _tune_history_keys(metric)
        for value in history.get(key, [])
        if not math.isnan(float(value))
    ]
    if not values:
        raise RuntimeError(f'No finite values found for tuning metric {metric!r}.')
    return max(values) if _tune_maximizes(metric) else min(values)


def _tune_history_keys(metric: str) -> tuple[str, str]:
    """
    Return live and SWA history keys for a validation tuning metric.
    """
    if metric == 'val_cosine':
        return ('val_cosine', 'swa_val_cosine')
    if metric == 'val_loss':
        return ('val_loss', 'swa_val_loss')
    raise ValueError(f'Unsupported tuning metric: {metric}')


def _tune_improved(value: float, best_value: float, metric: str) -> bool:
    """
    Return whether a tuning score improves the previous best.
    """
    return value >= best_value if _tune_maximizes(metric) else value <= best_value


def _tune_maximizes(metric: str) -> bool:
    """
    Return whether larger metric values are better.
    """
    return metric == 'val_cosine'


def _validation_tune_trial_dir(args: argparse.Namespace, output: Path) -> Path:
    """
    Return the directory used for per-trial tuning artifacts.
    """
    if args.tune_output_dir:
        return Path(args.tune_output_dir)
    return output.parent / f'{output.stem}_tune_trials'


def _history_path(checkpoint_path: Path) -> Path:
    """
    Match the training loop's history sidecar path for a checkpoint.
    """
    return Path(str(checkpoint_path).replace('.pt', '.history.json'))


def _tune_summary_path(output: Path) -> Path:
    """
    Return the JSON summary path for a validation tuning run.
    """
    return Path(str(output).replace('.pt', '.tune.json'))


def _cleanup_validation_tune_trials(
    trial_summaries: list[dict[str, Any]],
    trial_dir: Path,
) -> None:
    """
    Remove generated per-trial checkpoints when --no-tune-keep-trials is active.
    """
    for summary in trial_summaries:
        for key in ('checkpoint', 'history'):
            path = Path(str(summary[key]))
            if path.exists():
                path.unlink()
    try:
        trial_dir.rmdir()
    except OSError:
        pass


def _validate_loaded_checkpoint_config(
    model: MiraFragModel,
    *,
    mz_max: float,
    bin_width: float,
) -> None:
    """
    Validate runtime bin settings for a resumed checkpoint.

    The wrapper keeps the train CLI symmetric with evaluation and centralizes the checkpoint/bin compatibility check.
    """
    validate_checkpoint_bin_config(model, mz_max=mz_max, bin_width=bin_width)


def _apply_fragment_args_to_model_config(
    config: MiraFragConfig,
    args: argparse.Namespace,
) -> None:
    """
    Apply safe fragment candidate overrides to a loaded model config.

    This excludes head architecture fields so resuming from a checkpoint cannot accidentally change tensor shapes in the saved head.
    """
    apply_fragment_args_to_model_config(config, args)


def _mirafrag_config_value(value, field_name: str):
    """
    Return a CLI value or the MiraFragConfig default for a field.

    The helper is used while constructing new model configs from optional fragment arguments.
    """
    if value is not None:
        return value
    return MiraFragConfig.__dataclass_fields__[field_name].default


def _train_config(
    args: argparse.Namespace,
    *,
    fine_tune_strategy: str,
) -> dict[str, object]:
    """
    Serialize training command arguments for checkpoint metadata.

    The returned dictionary records the raw CLI settings plus the resolved fine-tune strategy for later inspection.
    """
    config = vars(args).copy()
    config['resolved_fine_tune_strategy'] = fine_tune_strategy
    config['head_delta_regularization'] = float(args.head_delta_regularization)
    config['encoder_delta_regularization'] = float(args.encoder_delta_regularization)
    return config


if __name__ == '__main__':
    main()
