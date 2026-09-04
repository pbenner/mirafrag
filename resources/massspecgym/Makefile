UV ?= uv
RESOURCE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(RESOURCE_DIR)/../..)
ENCODER ?= aimnet
CHECKPOINT_ENCODER ?= $(if $(filter unimol,$(strip $(ENCODER))),unimol,all)
PYTHON_EXTRA_ARGS = --extra $(ENCODER)
PYTHON ?= $(UV) run --project $(ROOT) $(PYTHON_EXTRA_ARGS) python
CHECKPOINT_PYTHON ?= $(UV) run --project $(ROOT) --extra $(CHECKPOINT_ENCODER) python
DEV_PYTHON ?= $(UV) run --project $(ROOT) --extra dev python
RUFF ?= $(DEV_PYTHON) -m ruff
PYTEST ?= $(DEV_PYTHON) -m pytest
PYTHONPATH := $(ROOT)/src

INPUT ?= $(RESOURCE_DIR)/data/MassSpecGym.tsv
PREDICT_INPUT ?= $(RESOURCE_DIR)/data/molecules.csv
MODEL ?= $(RESOURCE_DIR)/checkpoints/mirafrag.pt
INIT_MODEL ?=
DELTA_INIT_MODEL ?= $(RESOURCE_DIR)/checkpoints/mirafrag.pt
DELTA_MODEL ?= $(RESOURCE_DIR)/checkpoints/mirafrag_delta.pt
PREDICTIONS ?= $(RESOURCE_DIR)/predictions/massspecgym_test_predictions.csv
STRATIFIED_OUTPUT ?= $(RESOURCE_DIR)/predictions/massspecgym_stratified_eval.csv
PREDICT_OUTPUT ?= $(RESOURCE_DIR)/predictions/mirafrag_predictions.csv
ORACLE_OUTPUT ?= $(RESOURCE_DIR)/predictions/massspecgym_oracle.csv
RETRIEVAL_OUTPUT ?= $(RESOURCE_DIR)/predictions/massspecgym_retrieval.csv
GAP_DIAGNOSTIC_OUTPUT ?= $(RESOURCE_DIR)/predictions/massspecgym_gap_diagnostics.csv
GAP_DIAGNOSTIC_SUMMARY ?= $(RESOURCE_DIR)/predictions/massspecgym_gap_summary.csv
GAP_DIAGNOSTIC_PREDICTIONS ?= $(PREDICTIONS)
ENSEMBLE_OUTPUT ?= $(RESOURCE_DIR)/predictions/massspecgym_ensemble.csv
ENSEMBLE_SUMMARY ?= $(RESOURCE_DIR)/predictions/massspecgym_ensemble_summary.csv
ENSEMBLE_PREDICTIONS ?=
ENSEMBLE_NAMES ?=
ENSEMBLE_WEIGHTS_GRID ?= auto
RETRIEVAL_CANDIDATE_KIND ?= formula
RETRIEVAL_FORMULA_CANDIDATES ?= $(RESOURCE_DIR)/data/molecules/MassSpecGym_retrieval_candidates_formula.json
RETRIEVAL_MASS_CANDIDATES ?= $(RESOURCE_DIR)/data/molecules/MassSpecGym_retrieval_candidates_mass.json
RETRIEVAL_OFFICIAL_CANDIDATES ?= $(RESOURCE_DIR)/data/molecules/MassSpecGym_retrieval_candidates_$(RETRIEVAL_CANDIDATE_KIND).json
RETRIEVAL_CANDIDATES ?= $(RETRIEVAL_OFFICIAL_CANDIDATES)
DISK_CACHE_DIR ?= $(RESOURCE_DIR)/data/cache/mirafrag_features
CACHE_SOURCE_MODEL ?=
CACHE_SPLITS ?= train val test
CACHE_NUM_WORKERS ?= 40
CACHE_CHUNK_SIZE ?= 1
TRAIN_PREFILL_CACHE ?= 1
TRAIN_CACHE_NUM_WORKERS ?= $(NUM_WORKERS)
TRAIN_CACHE_CHUNK_SIZE ?= 1
ORACLE_CHUNK_SIZE ?= $(BATCH_SIZE)
RETRIEVAL_CANDIDATE_MODE ?= explicit
RETRIEVAL_MAX_CANDIDATES ?= 256
RETRIEVAL_QUERY_CHUNK_SIZE ?= 64
RETRIEVAL_HIT_KS ?= 1,5,10,20
RETRIEVAL_SCORE ?= cosine
RETRIEVAL_FAST ?= 1
RETRIEVAL_PREFILL_CACHE ?= 1
RETRIEVAL_MASS_TOLERANCE ?= 0.01
RETRIEVAL_CACHE_CHUNK_SIZE ?= $(CACHE_CHUNK_SIZE)
RETRIEVAL_SCORE_NUM_WORKERS ?= 0
RETRIEVAL_SCORE_SAMPLE_TIMEOUT ?= 300
RETRIEVAL_CACHE_TASK_TIMEOUT ?= 0
RETRIEVAL_CACHE_POOL_STALL_TIMEOUT ?= 0
RETRIEVAL_ROW_OFFSET ?= 0
RETRIEVAL_SAMPLE_ROWS ?=
RETRIEVAL_SAMPLE_SEED ?= 0
CE_BINS ?= 4
STRATIFY_MIN_COUNT ?= 25
EVAL_TOP_K ?= 100
EVAL_MIN_INTENSITY ?= 0.001
GAP_DIAGNOSTIC_MIN_COUNT ?= 25
GAP_DIAGNOSTIC_SIMILARITY_BINS ?= 0,0.3,0.5,0.7,0.85,0.95,1.0
GAP_DIAGNOSTIC_SPECTRUM_K ?= 0
GAP_DIAGNOSTIC_SPECTRUM_CE_WINDOW ?= 10
GAP_DIAGNOSTIC_SAME_INSTRUMENT ?= 1
GAP_DIAGNOSTIC_SAME_ADDUCT ?= 1
GAP_DIAGNOSTIC_RELAXED_FALLBACK ?= 1

DATASET_REPO ?= roman-bushuiev/MassSpecGym
DATASET_FILE ?= data/MassSpecGym.tsv
DATASET_REVISION ?= main
DATASET_LOCAL_DIR ?= $(RESOURCE_DIR)

FOUNDATION_SOURCE ?= off
FOUNDATION_MODEL ?= medium
FOUNDATION_PATH ?=
AIMNET_MODEL ?= aimnet2
AIMNET_PATH ?=
UNIMOL_MODEL_NAME ?= unimolv1
UNIMOL_MODEL_SIZE ?= 84m
UNIMOL_PRETRAINED_MODEL_PATH ?=
UNIMOL_PRETRAINED_DICT_PATH ?=
UNIMOL_MAX_ATOMS ?= 512
UNIMOL_MODE ?= trainable
GRAPH_RELAXATION ?=
AIMNET_RELAX_MODEL ?= $(AIMNET_MODEL)
AIMNET_RELAX_STEPS ?= 50
AIMNET_RELAX_FMAX ?= 0.05
AIMNET_RELAX_DEVICE ?= auto
COLLISION_ENERGY_MODE ?= raw
MERGE_GROUP_SPECTRA ?= 0
MERGE_GROUP_COLS ?= auto
DEVICE ?= auto

EPOCHS ?= 200
BATCH_SIZE ?= 48
LR ?= 1e-3
HEAD_LR ?=
ENCODER_LR ?=
WEIGHT_DECAY ?= 0.0
HEAD_WEIGHT_DECAY ?= 0.0
ENCODER_WEIGHT_DECAY ?=
ENCODER_LAYER_LR_DECAY ?= 1.0
SCHEDULER ?= exponential
SCHEDULER_INTERVAL ?= epoch
MIN_LR_RATIO ?= 0.0001
EXPONENTIAL_GAMMA ?= 0.85
PLATEAU_FACTOR ?= 0.5
PLATEAU_PATIENCE ?= 1
MZ_MAX ?= 1005
BIN_WIDTH ?= 0.01
LOSS ?= decoupled_kl
KL_WEIGHT ?= 0.7
COSINE_WEIGHT ?=
COVERAGE_WEIGHT ?= 0.1
TARGET_POWER ?= 1.0
ENTROPY_WEIGHT ?= 0.0
MASS_TOLERANCE ?= 0.01
RELATIVE_MASS_TOLERANCE ?= 0
MASS_TOLERANCE_MIN_MZ ?= 200
NUM_WORKERS ?= 16
MAX_ROWS ?=
PROGRESS ?= 1
CHECKPOINT_METRIC ?= val_loss
MEMORY_CACHE ?= 0
SLOW_SAMPLE_SECONDS ?= 0
DATALOADER_TIMEOUT ?= 0
TRACE_SAMPLES ?= 0
HIDDEN_DIM ?= 512
NUM_LAYERS ?= 2
DROPOUT ?= 0.01
METADATA_DIM ?= 32
MAX_FRAGMENT_TREE_DEPTH ?= 3
MAX_FRAGMENT_BROKEN_BONDS ?= 4
MAX_FRAGMENTS ?= 1024
MAX_FRAGMENT_EDGES ?= 4096
HIGH_CE_FRAGMENT_THRESHOLD ?=
HIGH_CE_MAX_FRAGMENT_TREE_DEPTH ?= 4
HIGH_CE_MAX_FRAGMENT_BROKEN_BONDS ?= 6
HIGH_CE_MAX_FRAGMENTS ?= 2048
HIGH_CE_MAX_FRAGMENT_EDGES ?= 8192
FRAGMENT_GNN_LAYERS ?= 2
FRAGMENT_PATH_LAYERS ?= 3
FRAGMENT_PATH_PRIMARY ?= 0
FRAGMENT_ACTION_PRIMARY_LAYERS ?=
BOND_BREAK_GEOMETRY_FEATURES ?=
INCLUDE_FRAGMENT_ISOTOPES ?= 1
FINE_TUNE_STRATEGY ?= full

FRAGMENT_ISOTOPE_THRESHOLD ?= 0.001
MAX_FRAGMENT_ISOTOPE_PEAKS ?= 1
OVERRIDE_FRAGMENT_CONFIG ?= 1


TRAIN_SPLIT ?= train
VAL_SPLIT ?= val
EVAL_SPLIT ?= test
SPLIT_COL ?= auto

FOUNDATION_ARGS = --encoder $(ENCODER)
ifeq ($(strip $(ENCODER)),aimnet)
FOUNDATION_ARGS += --aimnet-model $(AIMNET_MODEL)
ifneq ($(strip $(AIMNET_PATH)),)
FOUNDATION_ARGS += --aimnet-path $(AIMNET_PATH)
endif
else ifeq ($(strip $(ENCODER)),mace)
FOUNDATION_ARGS += --foundation-source $(FOUNDATION_SOURCE) --foundation-model $(FOUNDATION_MODEL)
ifneq ($(strip $(FOUNDATION_PATH)),)
FOUNDATION_ARGS += --foundation-path $(FOUNDATION_PATH)
endif
else ifeq ($(strip $(ENCODER)),unimol)
FOUNDATION_ARGS += --unimol-model-name $(UNIMOL_MODEL_NAME) --unimol-model-size $(UNIMOL_MODEL_SIZE) --unimol-max-atoms $(UNIMOL_MAX_ATOMS) --unimol-mode $(UNIMOL_MODE)
ifneq ($(strip $(UNIMOL_PRETRAINED_MODEL_PATH)),)
FOUNDATION_ARGS += --unimol-pretrained-model-path $(UNIMOL_PRETRAINED_MODEL_PATH)
endif
ifneq ($(strip $(UNIMOL_PRETRAINED_DICT_PATH)),)
FOUNDATION_ARGS += --unimol-pretrained-dict-path $(UNIMOL_PRETRAINED_DICT_PATH)
endif
endif

PROGRESS_ARGS =
ifeq ($(strip $(PROGRESS)),0)
PROGRESS_ARGS += --no-progress
endif

CACHE_ARGS =
ifeq ($(strip $(MEMORY_CACHE)),1)
CACHE_ARGS += --memory-cache
else
CACHE_ARGS += --no-memory-cache
endif
ifneq ($(strip $(DISK_CACHE_DIR)),)
CACHE_ARGS += --disk-cache-dir $(DISK_CACHE_DIR)
endif
TRAIN_CACHE_PREFILL_ARGS = --cache-num-workers $(TRAIN_CACHE_NUM_WORKERS) --cache-chunk-size $(TRAIN_CACHE_CHUNK_SIZE)
ifeq ($(strip $(TRAIN_PREFILL_CACHE)),1)
TRAIN_CACHE_PREFILL_ARGS += --prefill-cache
else
TRAIN_CACHE_PREFILL_ARGS += --no-prefill-cache
endif

ISOTOPE_FRAGMENT_ARGS =
ifeq ($(strip $(INCLUDE_FRAGMENT_ISOTOPES)),0)
ISOTOPE_FRAGMENT_ARGS += --no-include-fragment-isotopes
else
ISOTOPE_FRAGMENT_ARGS += --include-fragment-isotopes
endif

CACHE_USES_CHECKPOINT = 0
CACHE_MODEL_ARGS =
CACHE_PYTHON = $(PYTHON)
ifneq ($(strip $(CACHE_SOURCE_MODEL)),)
CACHE_USES_CHECKPOINT = 1
CACHE_MODEL_ARGS += --init-checkpoint $(CACHE_SOURCE_MODEL)
CACHE_PYTHON = $(CHECKPOINT_PYTHON)
else
CACHE_MODEL_ARGS += $(FOUNDATION_ARGS)
endif

TRAIN_PYTHON = $(PYTHON)
TRAIN_FOUNDATION_ARGS = $(FOUNDATION_ARGS)
ifneq ($(strip $(INIT_MODEL)),)
TRAIN_PYTHON = $(CHECKPOINT_PYTHON)
TRAIN_FOUNDATION_ARGS =
endif

GRAPH_RELAXATION_ARGS =
ifneq ($(strip $(GRAPH_RELAXATION)),)
GRAPH_RELAXATION_ARGS += --graph-relaxation $(GRAPH_RELAXATION) --aimnet-relax-model $(AIMNET_RELAX_MODEL) --aimnet-relax-steps $(AIMNET_RELAX_STEPS) --aimnet-relax-fmax $(AIMNET_RELAX_FMAX) --aimnet-relax-device $(AIMNET_RELAX_DEVICE)
endif
DATA_PREPROCESS_ARGS = --collision-energy-mode $(COLLISION_ENERGY_MODE)
ifeq ($(strip $(MERGE_GROUP_SPECTRA)),1)
DATA_PREPROCESS_ARGS += --merge-group-spectra --merge-group-cols $(MERGE_GROUP_COLS)
endif
COMMON_ARGS = --device $(DEVICE) --mz-max $(MZ_MAX) --bin-width $(BIN_WIDTH) --batch-size $(BATCH_SIZE) --num-workers $(NUM_WORKERS) --checkpoint-metric $(CHECKPOINT_METRIC) $(PROGRESS_ARGS)
SAMPLE_DEBUG_ARGS = --slow-sample-seconds $(SLOW_SAMPLE_SECONDS)
ifeq ($(strip $(TRACE_SAMPLES)),1)
SAMPLE_DEBUG_ARGS += --trace-samples
else
SAMPLE_DEBUG_ARGS += --no-trace-samples
endif
LOADER_DEBUG_ARGS = $(SAMPLE_DEBUG_ARGS) --dataloader-timeout $(DATALOADER_TIMEOUT)
PATH_PRIMARY_ARGS =
ifeq ($(strip $(FRAGMENT_PATH_PRIMARY)),1)
PATH_PRIMARY_ARGS += --fragment-path-primary
endif
ACTION_PRIMARY_ARGS =
ifneq ($(strip $(FRAGMENT_ACTION_PRIMARY_LAYERS)),)
ACTION_PRIMARY_ARGS += --fragment-action-primary-layers $(FRAGMENT_ACTION_PRIMARY_LAYERS)
endif
ifneq ($(strip $(BOND_BREAK_GEOMETRY_FEATURES)),)
ifeq ($(strip $(BOND_BREAK_GEOMETRY_FEATURES)),1)
ACTION_PRIMARY_ARGS += --bond-break-geometry-features
else ifeq ($(strip $(BOND_BREAK_GEOMETRY_FEATURES)),0)
ACTION_PRIMARY_ARGS += --no-bond-break-geometry-features
endif
endif
ARCH_ARGS = --hidden-dim $(HIDDEN_DIM) --num-layers $(NUM_LAYERS) --dropout $(DROPOUT) --metadata-dim $(METADATA_DIM) --fragment-gnn-layers $(FRAGMENT_GNN_LAYERS) --fragment-path-layers $(FRAGMENT_PATH_LAYERS) $(PATH_PRIMARY_ARGS) $(ACTION_PRIMARY_ARGS)
HIGH_CE_FRAGMENT_ARGS =
ifneq ($(strip $(HIGH_CE_FRAGMENT_THRESHOLD)),)
HIGH_CE_FRAGMENT_ARGS += --high-ce-fragment-threshold $(HIGH_CE_FRAGMENT_THRESHOLD) --high-ce-max-fragment-tree-depth $(HIGH_CE_MAX_FRAGMENT_TREE_DEPTH) --high-ce-max-fragment-broken-bonds $(HIGH_CE_MAX_FRAGMENT_BROKEN_BONDS) --high-ce-max-fragments $(HIGH_CE_MAX_FRAGMENTS) --high-ce-max-fragment-edges $(HIGH_CE_MAX_FRAGMENT_EDGES)
endif
BASE_FRAGMENT_ARGS = --max-fragment-tree-depth $(MAX_FRAGMENT_TREE_DEPTH) --max-fragment-broken-bonds $(MAX_FRAGMENT_BROKEN_BONDS) --max-fragments $(MAX_FRAGMENTS) --max-fragment-edges $(MAX_FRAGMENT_EDGES) $(ISOTOPE_FRAGMENT_ARGS) --fragment-isotope-threshold $(FRAGMENT_ISOTOPE_THRESHOLD) --max-fragment-isotope-peaks $(MAX_FRAGMENT_ISOTOPE_PEAKS)
FRAGMENT_ARGS = $(BASE_FRAGMENT_ARGS) $(HIGH_CE_FRAGMENT_ARGS)
HEAD_ARGS = $(ARCH_ARGS) $(FRAGMENT_ARGS)
CHECKPOINT_FRAGMENT_ARGS = $(HIGH_CE_FRAGMENT_ARGS) $(ACTION_PRIMARY_ARGS)
ifeq ($(strip $(OVERRIDE_FRAGMENT_CONFIG)),1)
CHECKPOINT_FRAGMENT_ARGS = $(FRAGMENT_ARGS) $(ACTION_PRIMARY_ARGS)
endif
CHECKPOINT_HEAD_ARGS = $(CHECKPOINT_FRAGMENT_ARGS) --fragment-path-layers $(FRAGMENT_PATH_LAYERS) $(PATH_PRIMARY_ARGS)
TRAIN_HEAD_ARGS = $(HEAD_ARGS)
ifneq ($(strip $(INIT_MODEL)),)
TRAIN_HEAD_ARGS = $(CHECKPOINT_HEAD_ARGS)
endif
CACHE_HEAD_ARGS = $(FRAGMENT_ARGS) $(ACTION_PRIMARY_ARGS)
ifeq ($(strip $(CACHE_USES_CHECKPOINT)),1)
CACHE_HEAD_ARGS = $(CHECKPOINT_FRAGMENT_ARGS)
endif
FINE_TUNE_ARGS = --fine-tune-strategy $(FINE_TUNE_STRATEGY)
INIT_ARGS =
ifneq ($(strip $(INIT_MODEL)),)
INIT_ARGS += --init-checkpoint $(INIT_MODEL)
endif
EFFECTIVE_KL_WEIGHT = $(if $(strip $(COSINE_WEIGHT)),$(shell python3 -c 'print(max(0.0, min(1.0, 1.0 - float("$(COSINE_WEIGHT)"))))'),$(KL_WEIGHT))
LOSS_ARGS = --loss $(LOSS) --kl-weight $(EFFECTIVE_KL_WEIGHT) --coverage-weight $(COVERAGE_WEIGHT) --target-power $(TARGET_POWER) --entropy-weight $(ENTROPY_WEIGHT)
ifeq ($(strip $(RELATIVE_MASS_TOLERANCE)),1)
LOSS_ARGS += --relative-mass-tolerance
else
LOSS_ARGS += --no-relative-mass-tolerance
endif
LOSS_ARGS += --mass-tolerance $(MASS_TOLERANCE) --mass-tolerance-min-mz $(MASS_TOLERANCE_MIN_MZ)
EVAL_EXPORT_ARGS = --top-k $(EVAL_TOP_K) --min-intensity $(EVAL_MIN_INTENSITY)
EVAL_DIAGNOSTIC_ARGS = --mass-tolerance $(MASS_TOLERANCE) --mass-tolerance-min-mz $(MASS_TOLERANCE_MIN_MZ)
ifeq ($(strip $(RELATIVE_MASS_TOLERANCE)),1)
EVAL_DIAGNOSTIC_ARGS += --relative-mass-tolerance
else
EVAL_DIAGNOSTIC_ARGS += --no-relative-mass-tolerance
endif
EVAL_STRATIFY_ARGS = --stratify-metadata --stratify-output $(STRATIFIED_OUTPUT) --ce-bins $(CE_BINS) --stratify-min-count $(STRATIFY_MIN_COUNT)
GAP_DIAGNOSTIC_SPECTRUM_ARGS =
ifneq ($(strip $(GAP_DIAGNOSTIC_SPECTRUM_K)),0)
GAP_DIAGNOSTIC_SPECTRUM_ARGS += --spectrum-neighbor-k $(GAP_DIAGNOSTIC_SPECTRUM_K) --spectrum-ce-window $(GAP_DIAGNOSTIC_SPECTRUM_CE_WINDOW) --mz-max $(MZ_MAX) --bin-width $(BIN_WIDTH)
ifeq ($(strip $(GAP_DIAGNOSTIC_SAME_INSTRUMENT)),0)
GAP_DIAGNOSTIC_SPECTRUM_ARGS += --no-spectrum-same-instrument
endif
ifeq ($(strip $(GAP_DIAGNOSTIC_SAME_ADDUCT)),0)
GAP_DIAGNOSTIC_SPECTRUM_ARGS += --no-spectrum-same-adduct
endif
ifeq ($(strip $(GAP_DIAGNOSTIC_RELAXED_FALLBACK)),0)
GAP_DIAGNOSTIC_SPECTRUM_ARGS += --no-spectrum-relaxed-fallback
endif
endif

ENSEMBLE_ARGS = --weights-grid $(ENSEMBLE_WEIGHTS_GRID)
ifneq ($(strip $(ENSEMBLE_NAMES)),)
ENSEMBLE_ARGS += --prediction-names $(ENSEMBLE_NAMES)
endif
RETRIEVAL_CACHE_ENV = MIRAFRAG_CACHE_TASK_TIMEOUT_SECONDS=$(RETRIEVAL_CACHE_TASK_TIMEOUT) MIRAFRAG_CACHE_POOL_STALL_SECONDS=$(RETRIEVAL_CACHE_POOL_STALL_TIMEOUT)
RETRIEVAL_CANDIDATE_ARGS = --candidate-mode $(RETRIEVAL_CANDIDATE_MODE) --max-candidates $(RETRIEVAL_MAX_CANDIDATES) --query-chunk-size $(RETRIEVAL_QUERY_CHUNK_SIZE) --hit-ks $(RETRIEVAL_HIT_KS) --score $(RETRIEVAL_SCORE) --mass-candidate-tolerance $(RETRIEVAL_MASS_TOLERANCE) --cache-chunk-size $(RETRIEVAL_CACHE_CHUNK_SIZE) --score-num-workers $(RETRIEVAL_SCORE_NUM_WORKERS) --score-sample-timeout $(RETRIEVAL_SCORE_SAMPLE_TIMEOUT) --row-offset $(RETRIEVAL_ROW_OFFSET) $(if $(strip $(RETRIEVAL_CANDIDATES)),--candidate-input $(RETRIEVAL_CANDIDATES),)
ifneq ($(strip $(RETRIEVAL_SAMPLE_ROWS)),)
RETRIEVAL_CANDIDATE_ARGS += --sample-rows $(RETRIEVAL_SAMPLE_ROWS) --sample-seed $(RETRIEVAL_SAMPLE_SEED)
endif
ifeq ($(strip $(RETRIEVAL_FAST)),0)
RETRIEVAL_CANDIDATE_ARGS += --no-fast-retrieval
endif
ifeq ($(strip $(RETRIEVAL_PREFILL_CACHE)),1)
RETRIEVAL_CANDIDATE_ARGS += --prefill-cache
else
RETRIEVAL_CANDIDATE_ARGS += --no-prefill-cache
endif
SCHEDULER_ARGS = --scheduler $(SCHEDULER) --min-lr-ratio $(MIN_LR_RATIO)
ifneq ($(filter $(strip $(SCHEDULER)),cosine constant exponential),)
SCHEDULER_ARGS += --scheduler-interval $(SCHEDULER_INTERVAL)
endif
ifeq ($(strip $(SCHEDULER)),exponential)
SCHEDULER_ARGS += --exponential-gamma $(EXPONENTIAL_GAMMA)
endif
ifeq ($(strip $(SCHEDULER)),plateau)
SCHEDULER_ARGS += --plateau-factor $(PLATEAU_FACTOR) --plateau-patience $(PLATEAU_PATIENCE)
endif
OPTIM_ARGS = --learning-rate $(LR) --weight-decay $(WEIGHT_DECAY) --head-weight-decay $(HEAD_WEIGHT_DECAY) --encoder-layer-lr-decay $(ENCODER_LAYER_LR_DECAY) $(SCHEDULER_ARGS)
ifneq ($(strip $(HEAD_LR)),)
OPTIM_ARGS += --head-learning-rate $(HEAD_LR)
endif
ifneq ($(strip $(ENCODER_LR)),)
OPTIM_ARGS += --encoder-learning-rate $(ENCODER_LR)
endif
ifneq ($(strip $(ENCODER_WEIGHT_DECAY)),)
OPTIM_ARGS += --encoder-weight-decay $(ENCODER_WEIGHT_DECAY)
endif
ROW_LIMIT_ARGS =
ifneq ($(strip $(MAX_ROWS)),)
ROW_LIMIT_ARGS += --max-rows $(MAX_ROWS)
endif

.PHONY: help install install-dev install-data-deps prepare-data prepare-retrieval-candidates prepare-cache train train-finetune eval eval-stratified gap-diagnostics ensemble-eval retrieval-eval oracle predict test lint format clean


help:
	@printf '%s\n' \
		'MiraFrag targets:' \
		'  make install             Sync the default uv environment' \
		'  make install-dev         Sync uv environment with dev tools' \
		'  make prepare-data        Download MassSpecGym TSV from Hugging Face' \
		'  make      Convert FIORA MSnLib library.csv to MiraFrag TSV' \
		'  make prepare-retrieval-candidates Download official retrieval candidates' \
		'  make prepare-cache       Precompute graph/fragment feature cache' \
		'  make train-finetune      Same as train, with explicit fine-tune target name' \
		'  make         Continue DELTA_INIT_MODEL with delta fine-tuning' \
		'  make eval INPUT=...      Evaluate checkpoint on MassSpecGym split' \
		'  make eval-stratified    Evaluate checkpoint and group metrics by metadata' \
		'  make gap-diagnostics    Analyze validation/test gap by train-nearest chemistry' \
		'  make ensemble-eval      Blend exported prediction CSVs and score spectra' \
		'  make retrieval-eval     Rank official MassSpecGym candidates and report Hit@k' \
		'  make oracle              Compute fragment-support oracle diagnostics' \
		'  make predict             Predict spectra for PREDICT_INPUT' \
		'  make test                Run smoke tests' \
		'  make lint                Run ruff check' \
		'' \
		'Common variables:' \
		'  INPUT=$(INPUT)' \
		'  MODEL=$(MODEL)' \
		'  INIT_MODEL=$(INIT_MODEL)' \
		'  DELTA_INIT_MODEL=$(DELTA_INIT_MODEL)' \
		'  DELTA_MODEL=$(DELTA_MODEL)' \
		'  DISK_CACHE_DIR=$(DISK_CACHE_DIR)' \
		'  CACHE_SOURCE_MODEL=$(CACHE_SOURCE_MODEL)' \
		'  CACHE_SPLITS=$(CACHE_SPLITS)' \
		'  CACHE_NUM_WORKERS=$(CACHE_NUM_WORKERS)' \
		'  CACHE_CHUNK_SIZE=$(CACHE_CHUNK_SIZE)' \
		'  TRAIN_PREFILL_CACHE=$(TRAIN_PREFILL_CACHE)' \
		'  TRAIN_CACHE_NUM_WORKERS=$(TRAIN_CACHE_NUM_WORKERS)' \
		'  TRAIN_CACHE_CHUNK_SIZE=$(TRAIN_CACHE_CHUNK_SIZE)' \
		'  ORACLE_CHUNK_SIZE=$(ORACLE_CHUNK_SIZE)' \
		'  RETRIEVAL_OUTPUT=$(RETRIEVAL_OUTPUT)' \
		'  STRATIFIED_OUTPUT=$(STRATIFIED_OUTPUT)' \
		'  GAP_DIAGNOSTIC_OUTPUT=$(GAP_DIAGNOSTIC_OUTPUT)' \
		'  GAP_DIAGNOSTIC_SUMMARY=$(GAP_DIAGNOSTIC_SUMMARY)' \
		'  RETRIEVAL_CANDIDATES=$(RETRIEVAL_CANDIDATES)' \
		'  RETRIEVAL_CANDIDATE_KIND=$(RETRIEVAL_CANDIDATE_KIND)' \
		'  RETRIEVAL_OFFICIAL_CANDIDATES=$(RETRIEVAL_OFFICIAL_CANDIDATES)' \
		'  RETRIEVAL_CANDIDATE_MODE=$(RETRIEVAL_CANDIDATE_MODE)' \
		'  RETRIEVAL_MAX_CANDIDATES=$(RETRIEVAL_MAX_CANDIDATES)' \
		'  RETRIEVAL_HIT_KS=$(RETRIEVAL_HIT_KS)' \
		'  RETRIEVAL_FAST=$(RETRIEVAL_FAST)' \
		'  RETRIEVAL_PREFILL_CACHE=$(RETRIEVAL_PREFILL_CACHE)' \
		'  RETRIEVAL_SCORE_NUM_WORKERS=$(RETRIEVAL_SCORE_NUM_WORKERS)' \
		'  RETRIEVAL_SCORE_SAMPLE_TIMEOUT=$(RETRIEVAL_SCORE_SAMPLE_TIMEOUT)' \
		'  RETRIEVAL_CACHE_TASK_TIMEOUT=$(RETRIEVAL_CACHE_TASK_TIMEOUT)' \
		'  RETRIEVAL_CACHE_POOL_STALL_TIMEOUT=$(RETRIEVAL_CACHE_POOL_STALL_TIMEOUT)' \
		'  RETRIEVAL_ROW_OFFSET=$(RETRIEVAL_ROW_OFFSET)' \
		'  RETRIEVAL_SAMPLE_ROWS=$(RETRIEVAL_SAMPLE_ROWS)' \
		'  RETRIEVAL_SAMPLE_SEED=$(RETRIEVAL_SAMPLE_SEED)' \
		'  ENCODER=$(ENCODER)' \
		'  CHECKPOINT_ENCODER=$(CHECKPOINT_ENCODER)' \
		'  FOUNDATION_SOURCE=$(FOUNDATION_SOURCE)' \
		'  FOUNDATION_MODEL=$(FOUNDATION_MODEL)' \
		'  AIMNET_MODEL=$(AIMNET_MODEL)' \
		'  GRAPH_RELAXATION=$(GRAPH_RELAXATION)' \
		'  AIMNET_RELAX_MODEL=$(AIMNET_RELAX_MODEL)' \
		'  AIMNET_RELAX_STEPS=$(AIMNET_RELAX_STEPS)' \
		'  AIMNET_RELAX_FMAX=$(AIMNET_RELAX_FMAX)' \
		'  AIMNET_RELAX_DEVICE=$(AIMNET_RELAX_DEVICE)' \
		'  DEVICE=$(DEVICE)' \
		'  COLLISION_ENERGY_MODE=$(COLLISION_ENERGY_MODE)' \
		'  MERGE_GROUP_SPECTRA=$(MERGE_GROUP_SPECTRA)' \
		'  MERGE_GROUP_COLS=$(MERGE_GROUP_COLS)' \
		'  EPOCHS=$(EPOCHS)' \
		'  BATCH_SIZE=$(BATCH_SIZE)' \
		'  LR=$(LR)' \
		'  HEAD_LR=$(HEAD_LR)' \
		'  ENCODER_LR=$(ENCODER_LR)' \
		'  WEIGHT_DECAY=$(WEIGHT_DECAY)' \
		'  HEAD_WEIGHT_DECAY=$(HEAD_WEIGHT_DECAY)' \
		'  ENCODER_WEIGHT_DECAY=$(ENCODER_WEIGHT_DECAY)' \
		'  ENCODER_LAYER_LR_DECAY=$(ENCODER_LAYER_LR_DECAY)' \
		'  SCHEDULER=$(SCHEDULER)' \
		'  LOSS=$(LOSS)' \
		'  KL_WEIGHT=$(KL_WEIGHT)' \
		'  COSINE_WEIGHT=$(COSINE_WEIGHT)' \
		'  TARGET_POWER=$(TARGET_POWER)' \
		'  ENTROPY_WEIGHT=$(ENTROPY_WEIGHT)' \
		'  MASS_TOLERANCE=$(MASS_TOLERANCE)' \
		'  RELATIVE_MASS_TOLERANCE=$(RELATIVE_MASS_TOLERANCE)' \
		'  MASS_TOLERANCE_MIN_MZ=$(MASS_TOLERANCE_MIN_MZ)' \
		'  CE_BINS=$(CE_BINS)' \
		'  STRATIFY_MIN_COUNT=$(STRATIFY_MIN_COUNT)' \
		'  SCHEDULER_INTERVAL=$(SCHEDULER_INTERVAL)' \
		'  MIN_LR_RATIO=$(MIN_LR_RATIO)' \
		'  EXPONENTIAL_GAMMA=$(EXPONENTIAL_GAMMA)' \
		'  PLATEAU_FACTOR=$(PLATEAU_FACTOR)' \
		'  PLATEAU_PATIENCE=$(PLATEAU_PATIENCE)' \
		'  NUM_WORKERS=$(NUM_WORKERS)' \
		'  SLOW_SAMPLE_SECONDS=$(SLOW_SAMPLE_SECONDS)' \
		'  DATALOADER_TIMEOUT=$(DATALOADER_TIMEOUT)' \
		'  TRACE_SAMPLES=$(TRACE_SAMPLES)' \
		'  MEMORY_CACHE=$(MEMORY_CACHE)' \
		'  HIDDEN_DIM=$(HIDDEN_DIM)' \
		'  NUM_LAYERS=$(NUM_LAYERS)' \
		'  DROPOUT=$(DROPOUT)' \
		'  METADATA_DIM=$(METADATA_DIM)' \
		'  FRAGMENT_PATH_LAYERS=$(FRAGMENT_PATH_LAYERS)' \
		'  FRAGMENT_PATH_PRIMARY=$(FRAGMENT_PATH_PRIMARY)' \
		'  MAX_FRAGMENT_TREE_DEPTH=$(MAX_FRAGMENT_TREE_DEPTH)' \
		'  MAX_FRAGMENT_BROKEN_BONDS=$(MAX_FRAGMENT_BROKEN_BONDS)' \
		'  HIGH_CE_FRAGMENT_THRESHOLD=$(HIGH_CE_FRAGMENT_THRESHOLD)' \
		'  HIGH_CE_MAX_FRAGMENT_TREE_DEPTH=$(HIGH_CE_MAX_FRAGMENT_TREE_DEPTH)' \
		'  HIGH_CE_MAX_FRAGMENT_BROKEN_BONDS=$(HIGH_CE_MAX_FRAGMENT_BROKEN_BONDS)' \
		'  HIGH_CE_MAX_FRAGMENTS=$(HIGH_CE_MAX_FRAGMENTS)' \
		'  HIGH_CE_MAX_FRAGMENT_EDGES=$(HIGH_CE_MAX_FRAGMENT_EDGES)' \
		'  MAX_FRAGMENTS=$(MAX_FRAGMENTS)' \
		'  OVERRIDE_FRAGMENT_CONFIG=$(OVERRIDE_FRAGMENT_CONFIG)' \
		'  CHECKPOINT_METRIC=$(CHECKPOINT_METRIC)' \
		'  PROGRESS=$(PROGRESS)'

install:
	$(UV) sync --project $(ROOT) --extra $(ENCODER)

install-dev:
	$(UV) sync --project $(ROOT) --extra dev

install-data-deps:
	$(UV) run --no-project --with huggingface_hub --with hf_xet python -c "import huggingface_hub"

prepare-data: install-data-deps
	mkdir -p $(DATASET_LOCAL_DIR)/$(dir $(DATASET_FILE))
	$(UV) run --no-project --with huggingface_hub --with hf_xet python -c "from huggingface_hub import hf_hub_download; path = hf_hub_download(repo_id='$(DATASET_REPO)', filename='$(DATASET_FILE)', repo_type='dataset', revision='$(DATASET_REVISION)', local_dir='$(DATASET_LOCAL_DIR)'); print(path)"

$(RESOURCE_DIR)/data/molecules/MassSpecGym_retrieval_candidates_%.json: install-data-deps
	mkdir -p $(dir $@)
	$(UV) run --no-project --with huggingface_hub --with hf_xet python -c "from huggingface_hub import hf_hub_download; path = hf_hub_download(repo_id='$(DATASET_REPO)', filename='data/molecules/MassSpecGym_retrieval_candidates_$*.json', repo_type='dataset', revision='$(DATASET_REVISION)', local_dir='$(DATASET_LOCAL_DIR)'); print(path)"

prepare-retrieval-candidates: $(RETRIEVAL_FORMULA_CANDIDATES) $(RETRIEVAL_MASS_CANDIDATES)

prepare-cache:
	mkdir -p $(DISK_CACHE_DIR)
	PYTHONPATH=$(PYTHONPATH) $(CACHE_PYTHON) -m mirafrag.cli.cache \
		--input $(INPUT) \
		--disk-cache-dir $(DISK_CACHE_DIR) \
		$(CACHE_MODEL_ARGS) \
		--device $(DEVICE) --mz-max $(MZ_MAX) --bin-width $(BIN_WIDTH) --chunk-size $(CACHE_CHUNK_SIZE) --num-workers $(CACHE_NUM_WORKERS) $(PROGRESS_ARGS) $(GRAPH_RELAXATION_ARGS) \
		$(SAMPLE_DEBUG_ARGS) \
		$(CACHE_HEAD_ARGS) \
		$(ROW_LIMIT_ARGS) \
		--splits $(CACHE_SPLITS) \
		--split-col $(SPLIT_COL)

train:
	mkdir -p $(dir $(MODEL))
	PYTHONPATH=$(PYTHONPATH) $(TRAIN_PYTHON) -m mirafrag.cli.train \
		--input $(INPUT) \
		--output $(MODEL) \
		$(INIT_ARGS) \
		$(TRAIN_FOUNDATION_ARGS) \
		$(COMMON_ARGS) \
		$(DATA_PREPROCESS_ARGS) \
		$(GRAPH_RELAXATION_ARGS) \
		$(LOADER_DEBUG_ARGS) \
		$(TRAIN_HEAD_ARGS) \
		$(CACHE_ARGS) \
		$(TRAIN_CACHE_PREFILL_ARGS) \
		$(ROW_LIMIT_ARGS) \
		--epochs $(EPOCHS) \
		$(LOSS_ARGS) \
		$(OPTIM_ARGS) \
		--train-split $(TRAIN_SPLIT) \
		--val-split $(VAL_SPLIT) \
		--split-col $(SPLIT_COL) \
		$(FINE_TUNE_ARGS)

train-finetune:
	mkdir -p $(dir $(MODEL))
	PYTHONPATH=$(PYTHONPATH) $(TRAIN_PYTHON) -m mirafrag.cli.train \
		--input $(INPUT) \
		--output $(MODEL) \
		$(INIT_ARGS) \
		$(TRAIN_FOUNDATION_ARGS) \
		$(COMMON_ARGS) \
		$(DATA_PREPROCESS_ARGS) \
		$(GRAPH_RELAXATION_ARGS) \
		$(LOADER_DEBUG_ARGS) \
		$(TRAIN_HEAD_ARGS) \
		$(CACHE_ARGS) \
		$(TRAIN_CACHE_PREFILL_ARGS) \
		$(ROW_LIMIT_ARGS) \
		--epochs $(EPOCHS) \
		$(LOSS_ARGS) \
		$(OPTIM_ARGS) \
		--train-split $(TRAIN_SPLIT) \
		--val-split $(VAL_SPLIT) \
		--split-col $(SPLIT_COL) \
		$(FINE_TUNE_ARGS)

eval:
	mkdir -p $(dir $(PREDICTIONS))
	PYTHONPATH=$(PYTHONPATH) $(CHECKPOINT_PYTHON) -m mirafrag.cli.eval \
		--input $(INPUT) \
		--model $(MODEL) \
		--output $(PREDICTIONS) \
		$(COMMON_ARGS) \
		$(GRAPH_RELAXATION_ARGS) \
		$(CACHE_ARGS) \
		$(EVAL_EXPORT_ARGS) \
		$(EVAL_DIAGNOSTIC_ARGS) \
		$(HIGH_CE_FRAGMENT_ARGS) \
		$(ROW_LIMIT_ARGS) \
		--split $(EVAL_SPLIT) \
		--split-col $(SPLIT_COL) \
		$(FINE_TUNE_ARGS)

eval-stratified:
	mkdir -p $(dir $(PREDICTIONS)) $(dir $(STRATIFIED_OUTPUT))
	PYTHONPATH=$(PYTHONPATH) $(CHECKPOINT_PYTHON) -m mirafrag.cli.eval \
		--input $(INPUT) \
		--model $(MODEL) \
		--output $(PREDICTIONS) \
		$(COMMON_ARGS) \
		$(GRAPH_RELAXATION_ARGS) \
		$(CACHE_ARGS) \
		$(EVAL_EXPORT_ARGS) \
		$(EVAL_DIAGNOSTIC_ARGS) \
		$(EVAL_STRATIFY_ARGS) \
		$(HIGH_CE_FRAGMENT_ARGS) \
		$(ROW_LIMIT_ARGS) \
		--split $(EVAL_SPLIT) \
		--split-col $(SPLIT_COL)

gap-diagnostics:
	mkdir -p $(dir $(GAP_DIAGNOSTIC_OUTPUT)) $(dir $(GAP_DIAGNOSTIC_SUMMARY))
	PYTHONPATH=$(PYTHONPATH) $(CHECKPOINT_PYTHON) -m mirafrag.cli.gap_diagnostics \
		--input $(INPUT) \
		--predictions $(GAP_DIAGNOSTIC_PREDICTIONS) \
		--output $(GAP_DIAGNOSTIC_OUTPUT) \
		--summary-output $(GAP_DIAGNOSTIC_SUMMARY) \
		--train-split $(TRAIN_SPLIT) \
		--eval-split $(EVAL_SPLIT) \
		--split-col $(SPLIT_COL) \
		--similarity-bins $(GAP_DIAGNOSTIC_SIMILARITY_BINS) \
		--min-count $(GAP_DIAGNOSTIC_MIN_COUNT) \
		$(GAP_DIAGNOSTIC_SPECTRUM_ARGS) \
		$(PROGRESS_ARGS) \
		$(ROW_LIMIT_ARGS)

ensemble-eval:
	mkdir -p $(dir $(ENSEMBLE_OUTPUT)) $(dir $(ENSEMBLE_SUMMARY))
	PYTHONPATH=$(PYTHONPATH) $(CHECKPOINT_PYTHON) -m mirafrag.cli.ensemble_eval \
		--input $(INPUT) \
		--predictions $(ENSEMBLE_PREDICTIONS) \
		--output $(ENSEMBLE_OUTPUT) \
		--summary-output $(ENSEMBLE_SUMMARY) \
		--mz-max $(MZ_MAX) --bin-width $(BIN_WIDTH) \
		--split $(EVAL_SPLIT) \
		--split-col $(SPLIT_COL) \
		$(ENSEMBLE_ARGS) \
		$(PROGRESS_ARGS) \
		$(ROW_LIMIT_ARGS)

retrieval-eval: $(RETRIEVAL_CANDIDATES)
	mkdir -p $(dir $(RETRIEVAL_OUTPUT))
	$(RETRIEVAL_CACHE_ENV) PYTHONPATH=$(PYTHONPATH) $(CHECKPOINT_PYTHON) -m mirafrag.cli.retrieval_eval \
		--input $(INPUT) \
		--model $(MODEL) \
		--output $(RETRIEVAL_OUTPUT) \
		$(COMMON_ARGS) \
		$(CACHE_ARGS) \
		$(RETRIEVAL_CANDIDATE_ARGS) \
		$(HIGH_CE_FRAGMENT_ARGS) \
		$(ROW_LIMIT_ARGS) \
		--split $(EVAL_SPLIT) \
		--split-col $(SPLIT_COL)

oracle:
	mkdir -p $(dir $(ORACLE_OUTPUT))
	PYTHONPATH=$(PYTHONPATH) $(CHECKPOINT_PYTHON) -m mirafrag.cli.oracle \
		--input $(INPUT) \
		--model $(MODEL) \
		--output $(ORACLE_OUTPUT) \
		--device $(DEVICE) --mz-max $(MZ_MAX) --bin-width $(BIN_WIDTH) --batch-size $(BATCH_SIZE) --chunk-size $(ORACLE_CHUNK_SIZE) --num-workers $(NUM_WORKERS) $(PROGRESS_ARGS) \
		$(CACHE_ARGS) \
		$(EVAL_DIAGNOSTIC_ARGS) \
		$(HIGH_CE_FRAGMENT_ARGS) \
		$(ROW_LIMIT_ARGS) \
		--split $(EVAL_SPLIT) \
		--split-col $(SPLIT_COL)

predict:
	mkdir -p $(dir $(PREDICT_OUTPUT))
	PYTHONPATH=$(PYTHONPATH) $(CHECKPOINT_PYTHON) -m mirafrag.cli.predict \
		--input $(PREDICT_INPUT) \
		--model $(MODEL) \
		--output $(PREDICT_OUTPUT) \
		$(COMMON_ARGS) \
		$(CACHE_ARGS) \
		$(HIGH_CE_FRAGMENT_ARGS)

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) -q $(ROOT)/tests

lint:
	$(RUFF) check $(ROOT)

format:
	$(RUFF) format $(ROOT)

clean:
	rm -rf $(ROOT)/.pytest_cache $(ROOT)/.ruff_cache $(ROOT)/src/mirafrag/__pycache__ $(ROOT)/src/mirafrag/cli/__pycache__ $(ROOT)/tests/__pycache__
