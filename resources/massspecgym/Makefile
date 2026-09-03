UV ?= uv
RESOURCE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(RESOURCE_DIR)/../..)
ENCODER ?= aimnet
CHECKPOINT_ENCODER ?= $(if $(filter unimol,$(strip $(ENCODER))),unimol,all)
PYTHON_EXTRA_ARGS = --extra $(ENCODER)
ifeq ($(strip $(ENCODER)),small3d)
PYTHON_EXTRA_ARGS =
endif
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
VALOPT_INIT_MODEL ?= $(RESOURCE_DIR)/checkpoints/mirafrag_aimnet_path.pt
VALOPT_MODEL ?= $(RESOURCE_DIR)/checkpoints/mirafrag_aimnet_valopt.pt
PREDICTIONS ?= $(RESOURCE_DIR)/predictions/massspecgym_test_predictions.csv
STRATIFIED_OUTPUT ?= $(RESOURCE_DIR)/predictions/massspecgym_stratified_eval.csv
PREDICT_OUTPUT ?= $(RESOURCE_DIR)/predictions/mirafrag_predictions.csv
ORACLE_OUTPUT ?= $(RESOURCE_DIR)/predictions/massspecgym_oracle.csv
RETRIEVAL_OUTPUT ?= $(RESOURCE_DIR)/predictions/massspecgym_retrieval.csv
GAP_DIAGNOSTIC_OUTPUT ?= $(RESOURCE_DIR)/predictions/massspecgym_gap_diagnostics.csv
GAP_DIAGNOSTIC_SUMMARY ?= $(RESOURCE_DIR)/predictions/massspecgym_gap_summary.csv
GAP_DIAGNOSTIC_PREDICTIONS ?= $(PREDICTIONS)
PLATEAU_DIAGNOSTIC_OUTPUT ?= $(RESOURCE_DIR)/predictions/massspecgym_plateau_diagnostics.csv
PLATEAU_DIAGNOSTIC_SUMMARY ?= $(RESOURCE_DIR)/predictions/massspecgym_plateau_summary.csv
PLATEAU_DIAGNOSTIC_MODELS ?= $(RESOURCE_DIR)/predictions/massspecgym_plateau_models.csv
PLATEAU_DIAGNOSTIC_WINNERS ?= $(RESOURCE_DIR)/predictions/massspecgym_plateau_winner_summary.csv
PLATEAU_DIAGNOSTIC_PREDICTIONS ?= $(PREDICTIONS)
PLATEAU_DIAGNOSTIC_NAMES ?=
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
RETRIEVAL_TRAIN_CANDIDATES ?=
RETRIEVAL_LOSS_WEIGHT ?= 0
RETRIEVAL_NEGATIVES ?= 16
RETRIEVAL_MAX_QUERIES ?=
DISTILL_PREDICTIONS ?=
DISTILL_WEIGHTS ?=
DISTILL_LOSS_WEIGHT ?= 0
DISTILL_MODE ?= fixed
DISTILL_ORACLE_TEMPERATURE ?= 0.03
DISTILL_ORACLE_MIN_TEACHER_WEIGHT ?= 0.05
DISTILL_MIN_OVERLAP_MASS ?= 0.05
DISTILL_CE_MIN ?=
DISTILL_CE_MAX ?=
RETRIEVAL_NUM_WORKERS ?=
RETRIEVAL_CACHE_NUM_WORKERS ?= $(RETRIEVAL_NUM_WORKERS)
RETRIEVAL_SAMPLE_TIMEOUT ?= 120
RETRIEVAL_CALIBRATION_HEAD ?= 0
CE_BINS ?= 4
STRATIFY_MIN_COUNT ?= 25
EVAL_TOP_K ?= 100
EVAL_MIN_INTENSITY ?= 0.001
GAP_DIAGNOSTIC_MIN_COUNT ?= 25
PLATEAU_DIAGNOSTIC_MIN_COUNT ?= 25
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
GROUP_BALANCED_LOSS ?= 0
GROUP_BALANCE_COLS ?= auto
GROUP_BALANCE_POWER ?= 1.0
DEVICE ?= auto
PHYSICAL_SIDECAR_MANIFEST ?= data/mirafrag_physical_sidecar_manifest.csv
PHYSICAL_SIDECAR_OUTPUT_DIR ?= data/physical_bond_features
PHYSICAL_SIDECAR_SPLITS ?= train val
PHYSICAL_SIDECAR_MODEL ?= $(MODEL)
PHYSICAL_SIDECAR_CID ?= cid-physical-features
PHYSICAL_SIDECAR_SKIP_GASTEIGER ?= 0
PHYSICAL_SIDECAR_EXPORT_CONFORMERS_DIR ?=

EPOCHS ?= 200
BATCH_SIZE ?= 48
LR ?= 1e-3
HEAD_LR ?=
ENCODER_LR ?=
WEIGHT_DECAY ?= 0.0
HEAD_WEIGHT_DECAY ?= 0.0
HEAD_DELTA_REGULARIZATION ?= 0.0
ENCODER_WEIGHT_DECAY ?=
ENCODER_LAYER_LR_DECAY ?= 1.0
ENCODER_DELTA_REGULARIZATION ?= 0.0
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
TARGET_NEIGHBOR_SMOOTHING ?= 0
TARGET_NEIGHBOR_WEIGHT ?= 0.3
TARGET_NEIGHBOR_CE_WINDOW ?= 10
TARGET_NEIGHBOR_SAME_INSTRUMENT ?= 1
TARGET_NEIGHBOR_SAME_ADDUCT ?= 1
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
FRAGMENT_ACTION_PATH_LAYERS ?=
FRAGMENT_ACTION_PRIMARY_LAYERS ?=
FRAGMENT_ACTION_PRIMARY_CE_GATE ?= 0
BOND_BREAK_GEOMETRY_FEATURES ?=
FRAGMENT_ACTION_GEOMETRY_FEATURES ?= 0
FRAGMENT_BOND_BREAK_LAYERS ?=
PHYSICAL_BOND_FEATURES ?= 0
PHYSICAL_BOND_FEATURES_PATH ?=
PHYSICAL_BOND_FEATURE_COLUMNS ?=
FRAGNNET_DAG_LAYERS ?=
FRAGNNET_DAG_NUM_HS ?=
CE_BASIS_FEATURES ?= 0
CE_EMBEDDING ?=
CE_FOURIER_FREQUENCIES ?= 8
AIMNET_CHARGE_FEATURES ?= 0
AIMNET_MULTIPASS_FEATURES ?= 0
ENCODER_BOND_ADAPTER_LAYERS ?=
ENCODER_BOND_ADAPTER_FEATURE_DIM ?= 128
ENCODER_BOND_ADAPTER_DROPOUT ?= 0.0
ENCODER_METADATA_ADAPTER_LAYERS ?=
ENCODER_METADATA_ADAPTER_FEATURE_DIM ?= 128
ENCODER_METADATA_ADAPTER_DROPOUT ?= 0.0
METADATA_CE_INTERACTION ?= 0
CANDIDATE_TRANSFORMER_LAYERS ?=
CANDIDATE_TRANSFORMER_HEADS ?= 8
CANDIDATE_TRANSFORMER_MAX_TOKENS ?= 256
CONDITIONAL_EXPERT_HEADS ?=
CONDITIONAL_EXPERT_HIDDEN_DIM ?= 128
CONDITIONAL_EXPERT_DROPOUT ?= 0.0
CANDIDATE_SUPPRESSION_GATE_HIDDEN_DIM ?=
CANDIDATE_SUPPRESSION_GATE_DROPOUT ?= 0.0
CANDIDATE_SUPPRESSION_GATE_INITIAL_PENALTY ?= 1e-6
MOLECULE_DESCRIPTOR_FEATURES ?= 0
MOLECULE_DESCRIPTOR_HIDDEN_DIM ?= 128
MOLECULE_DESCRIPTOR_DROPOUT ?= 0.0
INCLUDE_FRAGMENT_ISOTOPES ?= 1

FRAGMENT_ISOTOPE_THRESHOLD ?= 0.001
MAX_FRAGMENT_ISOTOPE_PEAKS ?= 1
FINE_TUNE_STRATEGY ?= full
OVERRIDE_FRAGMENT_CONFIG ?= 1

TUNE_METRIC ?= val_cosine
TUNE_TRIALS ?= 12
TUNE_EPOCHS ?= 12
TUNE_LRS ?= 1e-5,3e-5,1e-4
TUNE_HEAD_LRS ?=
TUNE_ENCODER_LRS ?=
TUNE_DROPOUTS ?= 0,0.02,0.05
TUNE_WEIGHT_DECAYS ?= 0,1e-6,1e-5
TUNE_HEAD_WEIGHT_DECAYS ?= 0
TUNE_ENCODER_WEIGHT_DECAYS ?=
TUNE_SWA_START_EPOCHS ?= 3,6
TUNE_SWA_LRS ?= 1e-5,3e-5,1e-4
TUNE_SEED ?=
TUNE_OUTPUT_DIR ?=
TUNE_KEEP_TRIALS ?= 0
VALOPT_LOSS ?= decoupled_kl
VALOPT_KL_WEIGHT ?= 0.7
VALOPT_COVERAGE_WEIGHT ?= 0.1
VALOPT_TARGET_POWER ?= 1.2
VALOPT_ENTROPY_WEIGHT ?= 0.0005
VALOPT_LR ?= 1e-5
VALOPT_WEIGHT_DECAY ?= 0.0

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
ifeq ($(strip $(GROUP_BALANCED_LOSS)),1)
DATA_PREPROCESS_ARGS += --group-balanced-loss --group-balance-cols $(GROUP_BALANCE_COLS) --group-balance-power $(GROUP_BALANCE_POWER)
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
BOND_BREAK_ARGS =
ifneq ($(strip $(FRAGMENT_BOND_BREAK_LAYERS)),)
BOND_BREAK_ARGS += --fragment-bond-break-layers $(FRAGMENT_BOND_BREAK_LAYERS)
endif
ACTION_PATH_ARGS =
ifneq ($(strip $(FRAGMENT_ACTION_PATH_LAYERS)),)
ACTION_PATH_ARGS += --fragment-action-path-layers $(FRAGMENT_ACTION_PATH_LAYERS)
endif
ACTION_PRIMARY_ARGS =
ifneq ($(strip $(FRAGMENT_ACTION_PRIMARY_LAYERS)),)
ACTION_PRIMARY_ARGS += --fragment-action-primary-layers $(FRAGMENT_ACTION_PRIMARY_LAYERS)
endif
ifeq ($(strip $(FRAGMENT_ACTION_PRIMARY_CE_GATE)),1)
ACTION_PRIMARY_ARGS += --fragment-action-primary-ce-gate
endif
ACTION_GEOMETRY_ARGS =
ifneq ($(strip $(BOND_BREAK_GEOMETRY_FEATURES)),)
ifeq ($(strip $(BOND_BREAK_GEOMETRY_FEATURES)),1)
ACTION_GEOMETRY_ARGS += --bond-break-geometry-features
else ifeq ($(strip $(BOND_BREAK_GEOMETRY_FEATURES)),0)
ACTION_GEOMETRY_ARGS += --no-bond-break-geometry-features
endif
else ifeq ($(strip $(FRAGMENT_ACTION_GEOMETRY_FEATURES)),1)
ACTION_GEOMETRY_ARGS += --fragment-action-geometry-features
endif
ACTION_PRIMARY_ARGS += $(ACTION_GEOMETRY_ARGS)
ifneq ($(strip $(FRAGMENT_ACTION_BOND_GNN_LAYERS)),)
ACTION_PRIMARY_ARGS += --fragment-action-bond-gnn-layers $(FRAGMENT_ACTION_BOND_GNN_LAYERS)
endif
PHYSICAL_BOND_FEATURE_ARGS =
ifeq ($(strip $(PHYSICAL_BOND_FEATURES)),1)
PHYSICAL_BOND_FEATURE_ARGS += --physical-bond-features --physical-bond-features-path $(PHYSICAL_BOND_FEATURES_PATH)
ifneq ($(strip $(PHYSICAL_BOND_FEATURE_COLUMNS)),)
PHYSICAL_BOND_FEATURE_ARGS += --physical-bond-feature-columns $(PHYSICAL_BOND_FEATURE_COLUMNS)
endif
endif
FRAGNNET_DAG_ARGS =
ifneq ($(strip $(FRAGNNET_DAG_LAYERS)),)
FRAGNNET_DAG_ARGS += --fragnnet-dag-layers $(FRAGNNET_DAG_LAYERS)
ifneq ($(strip $(FRAGNNET_DAG_NUM_HS)),)
FRAGNNET_DAG_ARGS += --fragnnet-dag-num-hs $(FRAGNNET_DAG_NUM_HS)
endif
endif
CE_BASIS_ARGS =
ifeq ($(strip $(CE_BASIS_FEATURES)),1)
CE_BASIS_ARGS += --ce-basis-features
endif
CE_EMBEDDING_ARGS =
ifneq ($(strip $(CE_EMBEDDING)),)
CE_EMBEDDING_ARGS += --ce-embedding $(CE_EMBEDDING) --ce-fourier-frequencies $(CE_FOURIER_FREQUENCIES)
endif
AIMNET_CHARGE_ARGS =
ifeq ($(strip $(AIMNET_CHARGE_FEATURES)),1)
AIMNET_CHARGE_ARGS += --aimnet-charge-features
endif
AIMNET_MULTIPASS_ARGS =
ifeq ($(strip $(AIMNET_MULTIPASS_FEATURES)),1)
AIMNET_MULTIPASS_ARGS += --aimnet-multipass-features
endif
ENCODER_BOND_ADAPTER_ARGS =
ifneq ($(strip $(ENCODER_BOND_ADAPTER_LAYERS)),)
ENCODER_BOND_ADAPTER_ARGS += --encoder-bond-adapter-layers $(ENCODER_BOND_ADAPTER_LAYERS) --encoder-bond-adapter-feature-dim $(ENCODER_BOND_ADAPTER_FEATURE_DIM) --encoder-bond-adapter-dropout $(ENCODER_BOND_ADAPTER_DROPOUT)
endif
ENCODER_METADATA_ADAPTER_ARGS =
ifneq ($(strip $(ENCODER_METADATA_ADAPTER_LAYERS)),)
ENCODER_METADATA_ADAPTER_ARGS += --encoder-metadata-adapter-layers $(ENCODER_METADATA_ADAPTER_LAYERS) --encoder-metadata-adapter-feature-dim $(ENCODER_METADATA_ADAPTER_FEATURE_DIM) --encoder-metadata-adapter-dropout $(ENCODER_METADATA_ADAPTER_DROPOUT)
endif
METADATA_CE_INTERACTION_ARGS =
ifeq ($(strip $(METADATA_CE_INTERACTION)),1)
METADATA_CE_INTERACTION_ARGS += --metadata-ce-interaction
endif
CANDIDATE_TRANSFORMER_ARGS =
ifneq ($(strip $(CANDIDATE_TRANSFORMER_LAYERS)),)
CANDIDATE_TRANSFORMER_ARGS += --candidate-transformer-layers $(CANDIDATE_TRANSFORMER_LAYERS) --candidate-transformer-heads $(CANDIDATE_TRANSFORMER_HEADS) --candidate-transformer-max-tokens $(CANDIDATE_TRANSFORMER_MAX_TOKENS)
endif
CANDIDATE_SUPPRESSION_GATE_ARGS =
ifneq ($(strip $(CANDIDATE_SUPPRESSION_GATE_HIDDEN_DIM)),)
CANDIDATE_SUPPRESSION_GATE_ARGS += --candidate-suppression-gate-hidden-dim $(CANDIDATE_SUPPRESSION_GATE_HIDDEN_DIM) --candidate-suppression-gate-dropout $(CANDIDATE_SUPPRESSION_GATE_DROPOUT) --candidate-suppression-gate-initial-penalty $(CANDIDATE_SUPPRESSION_GATE_INITIAL_PENALTY)
endif
CONDITIONAL_EXPERT_ARGS =
ifneq ($(strip $(CONDITIONAL_EXPERT_HEADS)),)
CONDITIONAL_EXPERT_ARGS += --conditional-expert-heads $(CONDITIONAL_EXPERT_HEADS) --conditional-expert-hidden-dim $(CONDITIONAL_EXPERT_HIDDEN_DIM) --conditional-expert-dropout $(CONDITIONAL_EXPERT_DROPOUT)
endif
MOLECULE_DESCRIPTOR_ARGS =
ifeq ($(strip $(MOLECULE_DESCRIPTOR_FEATURES)),1)
MOLECULE_DESCRIPTOR_ARGS += --molecule-descriptor-features --molecule-descriptor-hidden-dim $(MOLECULE_DESCRIPTOR_HIDDEN_DIM) --molecule-descriptor-dropout $(MOLECULE_DESCRIPTOR_DROPOUT)
endif
RETRIEVAL_CALIBRATION_ARGS =
ifeq ($(strip $(RETRIEVAL_CALIBRATION_HEAD)),1)
RETRIEVAL_CALIBRATION_ARGS += --retrieval-calibration-head
else
RETRIEVAL_CALIBRATION_ARGS += --no-retrieval-calibration-head
endif
ARCH_ARGS = --hidden-dim $(HIDDEN_DIM) --num-layers $(NUM_LAYERS) --dropout $(DROPOUT) --metadata-dim $(METADATA_DIM) --fragment-gnn-layers $(FRAGMENT_GNN_LAYERS) --fragment-path-layers $(FRAGMENT_PATH_LAYERS) $(PATH_PRIMARY_ARGS) $(ACTION_PATH_ARGS) $(ACTION_PRIMARY_ARGS) $(BOND_BREAK_ARGS) $(PHYSICAL_BOND_FEATURE_ARGS) $(FRAGNNET_DAG_ARGS) $(CE_BASIS_ARGS) $(CE_EMBEDDING_ARGS) $(AIMNET_CHARGE_ARGS) $(AIMNET_MULTIPASS_ARGS) $(ENCODER_BOND_ADAPTER_ARGS) $(ENCODER_METADATA_ADAPTER_ARGS) $(METADATA_CE_INTERACTION_ARGS) $(CANDIDATE_TRANSFORMER_ARGS) $(CANDIDATE_SUPPRESSION_GATE_ARGS) $(CONDITIONAL_EXPERT_ARGS) $(MOLECULE_DESCRIPTOR_ARGS) $(RETRIEVAL_CALIBRATION_ARGS)
HIGH_CE_FRAGMENT_ARGS =
ifneq ($(strip $(HIGH_CE_FRAGMENT_THRESHOLD)),)
HIGH_CE_FRAGMENT_ARGS += --high-ce-fragment-threshold $(HIGH_CE_FRAGMENT_THRESHOLD) --high-ce-max-fragment-tree-depth $(HIGH_CE_MAX_FRAGMENT_TREE_DEPTH) --high-ce-max-fragment-broken-bonds $(HIGH_CE_MAX_FRAGMENT_BROKEN_BONDS) --high-ce-max-fragments $(HIGH_CE_MAX_FRAGMENTS) --high-ce-max-fragment-edges $(HIGH_CE_MAX_FRAGMENT_EDGES)
endif
BASE_FRAGMENT_ARGS = --max-fragment-tree-depth $(MAX_FRAGMENT_TREE_DEPTH) --max-fragment-broken-bonds $(MAX_FRAGMENT_BROKEN_BONDS) --max-fragments $(MAX_FRAGMENTS) --max-fragment-edges $(MAX_FRAGMENT_EDGES) $(ISOTOPE_FRAGMENT_ARGS) --fragment-isotope-threshold $(FRAGMENT_ISOTOPE_THRESHOLD) --max-fragment-isotope-peaks $(MAX_FRAGMENT_ISOTOPE_PEAKS)
FRAGMENT_ARGS = $(BASE_FRAGMENT_ARGS) $(HIGH_CE_FRAGMENT_ARGS)
HEAD_ARGS = $(ARCH_ARGS) $(FRAGMENT_ARGS)
CHECKPOINT_FRAGMENT_ARGS = $(HIGH_CE_FRAGMENT_ARGS) $(BOND_BREAK_ARGS) $(ACTION_PRIMARY_ARGS)
ifeq ($(strip $(OVERRIDE_FRAGMENT_CONFIG)),1)
CHECKPOINT_FRAGMENT_ARGS = $(FRAGMENT_ARGS) $(BOND_BREAK_ARGS) $(ACTION_PRIMARY_ARGS) $(PHYSICAL_BOND_FEATURE_ARGS)
endif
CHECKPOINT_HEAD_ARGS = $(CHECKPOINT_FRAGMENT_ARGS) --fragment-path-layers $(FRAGMENT_PATH_LAYERS) $(PATH_PRIMARY_ARGS) $(ACTION_PATH_ARGS) $(FRAGNNET_DAG_ARGS) $(CE_BASIS_ARGS) $(CE_EMBEDDING_ARGS) $(AIMNET_CHARGE_ARGS) $(AIMNET_MULTIPASS_ARGS) $(ENCODER_BOND_ADAPTER_ARGS) $(ENCODER_METADATA_ADAPTER_ARGS) $(METADATA_CE_INTERACTION_ARGS) $(CANDIDATE_TRANSFORMER_ARGS) $(CANDIDATE_SUPPRESSION_GATE_ARGS) $(CONDITIONAL_EXPERT_ARGS) $(MOLECULE_DESCRIPTOR_ARGS) $(RETRIEVAL_CALIBRATION_ARGS)
TRAIN_HEAD_ARGS = $(HEAD_ARGS)
ifneq ($(strip $(INIT_MODEL)),)
TRAIN_HEAD_ARGS = $(CHECKPOINT_HEAD_ARGS)
endif
CACHE_HEAD_ARGS = $(FRAGMENT_ARGS) $(BOND_BREAK_ARGS) $(ACTION_PRIMARY_ARGS) $(PHYSICAL_BOND_FEATURE_ARGS)
ifeq ($(strip $(CACHE_USES_CHECKPOINT)),1)
CACHE_HEAD_ARGS = $(CHECKPOINT_FRAGMENT_ARGS)
endif
RETRIEVAL_TRAIN_ARGS = --retrieval-loss-weight $(RETRIEVAL_LOSS_WEIGHT) --retrieval-negatives $(RETRIEVAL_NEGATIVES) --retrieval-sample-timeout $(RETRIEVAL_SAMPLE_TIMEOUT) --retrieval-cache-chunk-size $(RETRIEVAL_CACHE_CHUNK_SIZE)
ifeq ($(strip $(RETRIEVAL_PREFILL_CACHE)),1)
RETRIEVAL_TRAIN_ARGS += --retrieval-prefill-cache
else
RETRIEVAL_TRAIN_ARGS += --no-retrieval-prefill-cache
endif
ifneq ($(strip $(RETRIEVAL_TRAIN_CANDIDATES)),)
RETRIEVAL_TRAIN_ARGS += --retrieval-candidate-input $(RETRIEVAL_TRAIN_CANDIDATES)
endif
ifneq ($(strip $(RETRIEVAL_MAX_QUERIES)),)
RETRIEVAL_TRAIN_ARGS += --retrieval-max-queries $(RETRIEVAL_MAX_QUERIES)
endif
ifneq ($(strip $(RETRIEVAL_NUM_WORKERS)),)
RETRIEVAL_TRAIN_ARGS += --retrieval-num-workers $(RETRIEVAL_NUM_WORKERS)
endif
ifneq ($(strip $(RETRIEVAL_CACHE_NUM_WORKERS)),)
RETRIEVAL_TRAIN_ARGS += --retrieval-cache-num-workers $(RETRIEVAL_CACHE_NUM_WORKERS)
endif
DISTILL_ARGS = --distill-loss-weight $(DISTILL_LOSS_WEIGHT) --distill-mode $(DISTILL_MODE) --distill-oracle-temperature $(DISTILL_ORACLE_TEMPERATURE) --distill-oracle-min-teacher-weight $(DISTILL_ORACLE_MIN_TEACHER_WEIGHT) --distill-min-overlap-mass $(DISTILL_MIN_OVERLAP_MASS)
ifneq ($(strip $(DISTILL_PREDICTIONS)),)
DISTILL_ARGS += --distill-predictions $(DISTILL_PREDICTIONS)
endif
ifneq ($(strip $(DISTILL_WEIGHTS)),)
DISTILL_ARGS += --distill-weights $(DISTILL_WEIGHTS)
endif
ifneq ($(strip $(DISTILL_CE_MIN)),)
DISTILL_ARGS += --distill-ce-min $(DISTILL_CE_MIN)
endif
ifneq ($(strip $(DISTILL_CE_MAX)),)
DISTILL_ARGS += --distill-ce-max $(DISTILL_CE_MAX)
endif
FINE_TUNE_ARGS = --fine-tune-strategy $(FINE_TUNE_STRATEGY)
INIT_ARGS =
ifneq ($(strip $(INIT_MODEL)),)
INIT_ARGS += --init-checkpoint $(INIT_MODEL)
endif
EFFECTIVE_KL_WEIGHT = $(if $(strip $(COSINE_WEIGHT)),$(shell python3 -c 'print(max(0.0, min(1.0, 1.0 - float("$(COSINE_WEIGHT)"))))'),$(KL_WEIGHT))
LOSS_ARGS = --loss $(LOSS) --kl-weight $(EFFECTIVE_KL_WEIGHT) --coverage-weight $(COVERAGE_WEIGHT) --target-power $(TARGET_POWER) --entropy-weight $(ENTROPY_WEIGHT)
ifeq ($(strip $(TARGET_NEIGHBOR_SMOOTHING)),1)
LOSS_ARGS += --target-neighbor-smoothing --target-neighbor-weight $(TARGET_NEIGHBOR_WEIGHT) --target-neighbor-ce-window $(TARGET_NEIGHBOR_CE_WINDOW)
ifeq ($(strip $(TARGET_NEIGHBOR_SAME_INSTRUMENT)),1)
LOSS_ARGS += --target-neighbor-same-instrument
else
LOSS_ARGS += --no-target-neighbor-same-instrument
endif
ifeq ($(strip $(TARGET_NEIGHBOR_SAME_ADDUCT)),1)
LOSS_ARGS += --target-neighbor-same-adduct
else
LOSS_ARGS += --no-target-neighbor-same-adduct
endif
endif
ifeq ($(strip $(RELATIVE_MASS_TOLERANCE)),1)
LOSS_ARGS += --relative-mass-tolerance
else
LOSS_ARGS += --no-relative-mass-tolerance
endif
LOSS_ARGS += --mass-tolerance $(MASS_TOLERANCE) --mass-tolerance-min-mz $(MASS_TOLERANCE_MIN_MZ)
VALOPT_LOSS_ARGS = --loss $(VALOPT_LOSS) --kl-weight $(VALOPT_KL_WEIGHT) --coverage-weight $(VALOPT_COVERAGE_WEIGHT) --target-power $(VALOPT_TARGET_POWER) --entropy-weight $(VALOPT_ENTROPY_WEIGHT)
ifeq ($(strip $(RELATIVE_MASS_TOLERANCE)),1)
VALOPT_LOSS_ARGS += --relative-mass-tolerance
else
VALOPT_LOSS_ARGS += --no-relative-mass-tolerance
endif
VALOPT_LOSS_ARGS += --mass-tolerance $(MASS_TOLERANCE) --mass-tolerance-min-mz $(MASS_TOLERANCE_MIN_MZ)
EVAL_EXPORT_ARGS = --top-k $(EVAL_TOP_K) --min-intensity $(EVAL_MIN_INTENSITY)
EVAL_DIAGNOSTIC_ARGS = --mass-tolerance $(MASS_TOLERANCE) --mass-tolerance-min-mz $(MASS_TOLERANCE_MIN_MZ)
ifeq ($(strip $(RELATIVE_MASS_TOLERANCE)),1)
EVAL_DIAGNOSTIC_ARGS += --relative-mass-tolerance
else
EVAL_DIAGNOSTIC_ARGS += --no-relative-mass-tolerance
endif
EVAL_PHYSICAL_BOND_FEATURE_ARGS =
ifneq ($(strip $(PHYSICAL_BOND_FEATURES_PATH)),)
EVAL_PHYSICAL_BOND_FEATURE_ARGS += --physical-bond-features-path $(PHYSICAL_BOND_FEATURES_PATH)
ifneq ($(strip $(PHYSICAL_BOND_FEATURE_COLUMNS)),)
EVAL_PHYSICAL_BOND_FEATURE_ARGS += --physical-bond-feature-columns $(PHYSICAL_BOND_FEATURE_COLUMNS)
endif
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
PLATEAU_DIAGNOSTIC_ARGS =
ifneq ($(strip $(PLATEAU_DIAGNOSTIC_NAMES)),)
PLATEAU_DIAGNOSTIC_ARGS += --prediction-names $(PLATEAU_DIAGNOSTIC_NAMES)
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
OPTIM_ARGS = --learning-rate $(LR) --weight-decay $(WEIGHT_DECAY) --head-weight-decay $(HEAD_WEIGHT_DECAY) --head-delta-regularization $(HEAD_DELTA_REGULARIZATION) --encoder-layer-lr-decay $(ENCODER_LAYER_LR_DECAY) --encoder-delta-regularization $(ENCODER_DELTA_REGULARIZATION) $(SCHEDULER_ARGS)
ifneq ($(strip $(HEAD_LR)),)
OPTIM_ARGS += --head-learning-rate $(HEAD_LR)
endif
ifneq ($(strip $(ENCODER_LR)),)
OPTIM_ARGS += --encoder-learning-rate $(ENCODER_LR)
endif
ifneq ($(strip $(ENCODER_WEIGHT_DECAY)),)
OPTIM_ARGS += --encoder-weight-decay $(ENCODER_WEIGHT_DECAY)
endif
TUNE_ARGS = --validation-tune --tune-metric $(TUNE_METRIC) --tune-trials $(TUNE_TRIALS) --tune-epochs $(TUNE_EPOCHS) --tune-lrs $(TUNE_LRS) --tune-dropouts $(TUNE_DROPOUTS) --tune-weight-decays $(TUNE_WEIGHT_DECAYS) --tune-head-weight-decays $(TUNE_HEAD_WEIGHT_DECAYS) --tune-swa-start-epochs $(TUNE_SWA_START_EPOCHS) --tune-swa-lrs $(TUNE_SWA_LRS)
ifneq ($(strip $(TUNE_HEAD_LRS)),)
TUNE_ARGS += --tune-head-lrs $(TUNE_HEAD_LRS)
endif
ifneq ($(strip $(TUNE_ENCODER_LRS)),)
TUNE_ARGS += --tune-encoder-lrs $(TUNE_ENCODER_LRS)
endif
ifneq ($(strip $(TUNE_ENCODER_WEIGHT_DECAYS)),)
TUNE_ARGS += --tune-encoder-weight-decays $(TUNE_ENCODER_WEIGHT_DECAYS)
endif
ifneq ($(strip $(TUNE_SEED)),)
TUNE_ARGS += --tune-seed $(TUNE_SEED)
endif
ifneq ($(strip $(TUNE_OUTPUT_DIR)),)
TUNE_ARGS += --tune-output-dir $(TUNE_OUTPUT_DIR)
endif
ifeq ($(strip $(TUNE_KEEP_TRIALS)),1)
TUNE_ARGS += --tune-keep-trials
else
TUNE_ARGS += --no-tune-keep-trials
endif

ROW_LIMIT_ARGS =
ifneq ($(strip $(MAX_ROWS)),)
ROW_LIMIT_ARGS += --max-rows $(MAX_ROWS)
endif

PHYSICAL_SIDECAR_CID_ARGS = --manifest $(PHYSICAL_SIDECAR_MANIFEST) --input-format mirafrag-graph --all-smiles-bonds --skip-xtb --skip-mace --output-dir $(PHYSICAL_SIDECAR_OUTPUT_DIR)
ifeq ($(strip $(PHYSICAL_SIDECAR_SKIP_GASTEIGER)),1)
PHYSICAL_SIDECAR_CID_ARGS += --skip-gasteiger
endif
ifneq ($(strip $(PHYSICAL_SIDECAR_EXPORT_CONFORMERS_DIR)),)
PHYSICAL_SIDECAR_CID_ARGS += --export-conformers-dir $(PHYSICAL_SIDECAR_EXPORT_CONFORMERS_DIR)
endif

.PHONY: help install install-dev install-data-deps prepare-data prepare-retrieval-candidates prepare-cache prepare-physical-sidecar-manifest physical-sidecar train train-finetune train-delta train-valopt eval eval-stratified gap-diagnostics plateau-diagnostics ensemble-eval retrieval-eval oracle predict test lint format clean

help:
	@printf '%s\n' \
		'MiraFrag targets:' \
		'  make install             Sync the default uv environment' \
		'  make install-dev         Sync uv environment with dev tools' \
		'  make prepare-data        Download MassSpecGym TSV from Hugging Face' \
		'  make      Convert FIORA MSnLib library.csv to MiraFrag TSV' \
		'  make prepare-retrieval-candidates Download official retrieval candidates' \
		'  make prepare-cache       Precompute graph/fragment feature cache' \
		'  make prepare-physical-sidecar-manifest Build graph-cache manifest for cid-physical-features' \
		'  make physical-sidecar    Generate RDKit/geometry physical bond sidecar from manifest' \
		'  make train INPUT=...     Train spectrum model using FINE_TUNE_STRATEGY' \
		'  make train-finetune      Same as train, with explicit fine-tune target name' \
		'  make train-delta         Continue DELTA_INIT_MODEL with delta fine-tuning' \
		'  make train-valopt        Tune validation hyperparameters from VALOPT_INIT_MODEL' \
		'  make eval INPUT=...      Evaluate checkpoint on MassSpecGym split' \
		'  make eval-stratified    Evaluate checkpoint and group metrics by metadata' \
		'  make gap-diagnostics    Analyze validation/test gap by train-nearest chemistry' \
		'  make plateau-diagnostics Explain score plateau from prediction CSVs' \
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
		'  VALOPT_INIT_MODEL=$(VALOPT_INIT_MODEL)' \
		'  VALOPT_MODEL=$(VALOPT_MODEL)' \
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
		'  PLATEAU_DIAGNOSTIC_OUTPUT=$(PLATEAU_DIAGNOSTIC_OUTPUT)' \
		'  PLATEAU_DIAGNOSTIC_SUMMARY=$(PLATEAU_DIAGNOSTIC_SUMMARY)' \
		'  PLATEAU_DIAGNOSTIC_MODELS=$(PLATEAU_DIAGNOSTIC_MODELS)' \
		'  PLATEAU_DIAGNOSTIC_WINNERS=$(PLATEAU_DIAGNOSTIC_WINNERS)' \
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
		'  RETRIEVAL_TRAIN_CANDIDATES=$(RETRIEVAL_TRAIN_CANDIDATES)' \
		'  RETRIEVAL_LOSS_WEIGHT=$(RETRIEVAL_LOSS_WEIGHT)' \
		'  RETRIEVAL_NEGATIVES=$(RETRIEVAL_NEGATIVES)' \
		'  RETRIEVAL_MAX_QUERIES=$(RETRIEVAL_MAX_QUERIES)' \
		'  DISTILL_PREDICTIONS=$(DISTILL_PREDICTIONS)' \
		'  DISTILL_WEIGHTS=$(DISTILL_WEIGHTS)' \
		'  DISTILL_LOSS_WEIGHT=$(DISTILL_LOSS_WEIGHT)' \
		'  DISTILL_MIN_OVERLAP_MASS=$(DISTILL_MIN_OVERLAP_MASS)' \
		'  RETRIEVAL_NUM_WORKERS=$(RETRIEVAL_NUM_WORKERS)' \
		'  RETRIEVAL_CACHE_NUM_WORKERS=$(RETRIEVAL_CACHE_NUM_WORKERS)' \
		'  RETRIEVAL_SAMPLE_TIMEOUT=$(RETRIEVAL_SAMPLE_TIMEOUT)' \
		'  RETRIEVAL_CALIBRATION_HEAD=$(RETRIEVAL_CALIBRATION_HEAD)' \
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
		'  HEAD_DELTA_REGULARIZATION=$(HEAD_DELTA_REGULARIZATION)' \
		'  ENCODER_WEIGHT_DECAY=$(ENCODER_WEIGHT_DECAY)' \
		'  ENCODER_LAYER_LR_DECAY=$(ENCODER_LAYER_LR_DECAY)' \
		'  ENCODER_DELTA_REGULARIZATION=$(ENCODER_DELTA_REGULARIZATION)' \
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
		'  AIMNET_CHARGE_FEATURES=$(AIMNET_CHARGE_FEATURES)' \
		'  AIMNET_MULTIPASS_FEATURES=$(AIMNET_MULTIPASS_FEATURES)' \
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
		'  PHYSICAL_BOND_FEATURES=$(PHYSICAL_BOND_FEATURES)' \
		'  MOLECULE_DESCRIPTOR_FEATURES=$(MOLECULE_DESCRIPTOR_FEATURES)' \
		'  PHYSICAL_BOND_FEATURES_PATH=$(PHYSICAL_BOND_FEATURES_PATH)' \
		'  PHYSICAL_SIDECAR_MANIFEST=$(PHYSICAL_SIDECAR_MANIFEST)' \
		'  PHYSICAL_SIDECAR_OUTPUT_DIR=$(PHYSICAL_SIDECAR_OUTPUT_DIR)' \
		'  FRAGMENT_ACTION_PATH_LAYERS=$(FRAGMENT_ACTION_PATH_LAYERS)' \
		'  FRAGNNET_DAG_LAYERS=$(FRAGNNET_DAG_LAYERS)' \
		'  FRAGNNET_DAG_NUM_HS=$(FRAGNNET_DAG_NUM_HS)' \
		'  FRAGMENT_BOND_BREAK_LAYERS=$(FRAGMENT_BOND_BREAK_LAYERS)' \
		'  MAX_FRAGMENT_TREE_DEPTH=$(MAX_FRAGMENT_TREE_DEPTH)' \
		'  MAX_FRAGMENT_BROKEN_BONDS=$(MAX_FRAGMENT_BROKEN_BONDS)' \
		'  HIGH_CE_FRAGMENT_THRESHOLD=$(HIGH_CE_FRAGMENT_THRESHOLD)' \
		'  HIGH_CE_MAX_FRAGMENT_TREE_DEPTH=$(HIGH_CE_MAX_FRAGMENT_TREE_DEPTH)' \
		'  HIGH_CE_MAX_FRAGMENT_BROKEN_BONDS=$(HIGH_CE_MAX_FRAGMENT_BROKEN_BONDS)' \
		'  HIGH_CE_MAX_FRAGMENTS=$(HIGH_CE_MAX_FRAGMENTS)' \
		'  HIGH_CE_MAX_FRAGMENT_EDGES=$(HIGH_CE_MAX_FRAGMENT_EDGES)' \
		'  MAX_FRAGMENTS=$(MAX_FRAGMENTS)' \
		'  OVERRIDE_FRAGMENT_CONFIG=$(OVERRIDE_FRAGMENT_CONFIG)' \
		'  FINE_TUNE_STRATEGY=$(FINE_TUNE_STRATEGY)' \
		'  CHECKPOINT_METRIC=$(CHECKPOINT_METRIC)' \
		'  TUNE_METRIC=$(TUNE_METRIC)' \
		'  TUNE_TRIALS=$(TUNE_TRIALS)' \
		'  TUNE_EPOCHS=$(TUNE_EPOCHS)' \
		'  TUNE_LRS=$(TUNE_LRS)' \
		'  TUNE_HEAD_LRS=$(TUNE_HEAD_LRS)' \
		'  TUNE_ENCODER_LRS=$(TUNE_ENCODER_LRS)' \
		'  TUNE_DROPOUTS=$(TUNE_DROPOUTS)' \
		'  TUNE_WEIGHT_DECAYS=$(TUNE_WEIGHT_DECAYS)' \
		'  TUNE_HEAD_WEIGHT_DECAYS=$(TUNE_HEAD_WEIGHT_DECAYS)' \
		'  TUNE_ENCODER_WEIGHT_DECAYS=$(TUNE_ENCODER_WEIGHT_DECAYS)' \
		'  VALOPT_LOSS=$(VALOPT_LOSS)' \
		'  VALOPT_TARGET_POWER=$(VALOPT_TARGET_POWER)' \
		'  VALOPT_ENTROPY_WEIGHT=$(VALOPT_ENTROPY_WEIGHT)' \
		'  VALOPT_LR=$(VALOPT_LR)' \
		'  VALOPT_WEIGHT_DECAY=$(VALOPT_WEIGHT_DECAY)' \
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

prepare-physical-sidecar-manifest:
	mkdir -p $(dir $(PHYSICAL_SIDECAR_MANIFEST))
	PYTHONPATH=$(PYTHONPATH) $(CHECKPOINT_PYTHON) -m mirafrag.cli.physical_sidecar_manifest \
		--input $(INPUT) \
		--model $(PHYSICAL_SIDECAR_MODEL) \
		--output $(PHYSICAL_SIDECAR_MANIFEST) \
		--disk-cache-dir $(DISK_CACHE_DIR) \
		--device cpu \
		--splits $(PHYSICAL_SIDECAR_SPLITS) \
		--split-col $(SPLIT_COL) \
		$(ROW_LIMIT_ARGS)

physical-sidecar: prepare-physical-sidecar-manifest
	mkdir -p $(PHYSICAL_SIDECAR_OUTPUT_DIR)
	$(PHYSICAL_SIDECAR_CID) $(PHYSICAL_SIDECAR_CID_ARGS)

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
		$(RETRIEVAL_TRAIN_ARGS) \
		$(DISTILL_ARGS) \
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

train-delta:
	$(MAKE) train-finetune MODEL=$(DELTA_MODEL) INIT_MODEL=$(DELTA_INIT_MODEL) FINE_TUNE_STRATEGY=delta

train-valopt: CHECKPOINT_METRIC = $(TUNE_METRIC)
train-valopt: LR = $(VALOPT_LR)
train-valopt: WEIGHT_DECAY = $(VALOPT_WEIGHT_DECAY)
train-valopt:
	mkdir -p $(dir $(VALOPT_MODEL))
	PYTHONPATH=$(PYTHONPATH) $(CHECKPOINT_PYTHON) -m mirafrag.cli.train \
		--input $(INPUT) \
		--output $(VALOPT_MODEL) \
		--init-checkpoint $(VALOPT_INIT_MODEL) \
		$(COMMON_ARGS) \
		$(DATA_PREPROCESS_ARGS) \
		$(GRAPH_RELAXATION_ARGS) \
		$(LOADER_DEBUG_ARGS) \
		$(CHECKPOINT_HEAD_ARGS) \
		$(CACHE_ARGS) \
		$(TRAIN_CACHE_PREFILL_ARGS) \
		$(ROW_LIMIT_ARGS) \
		--epochs $(EPOCHS) \
		$(VALOPT_LOSS_ARGS) \
		$(OPTIM_ARGS) \
		$(TUNE_ARGS) \
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
		$(EVAL_PHYSICAL_BOND_FEATURE_ARGS) \
		$(ROW_LIMIT_ARGS) \
		--split $(EVAL_SPLIT) \
		--split-col $(SPLIT_COL)

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
		$(EVAL_PHYSICAL_BOND_FEATURE_ARGS) \
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

plateau-diagnostics:
	mkdir -p $(dir $(PLATEAU_DIAGNOSTIC_OUTPUT)) $(dir $(PLATEAU_DIAGNOSTIC_SUMMARY)) $(dir $(PLATEAU_DIAGNOSTIC_MODELS)) $(dir $(PLATEAU_DIAGNOSTIC_WINNERS))
	PYTHONPATH=$(PYTHONPATH) $(CHECKPOINT_PYTHON) -m mirafrag.cli.plateau_diagnostics \
		--input $(INPUT) \
		--predictions $(PLATEAU_DIAGNOSTIC_PREDICTIONS) \
		--output $(PLATEAU_DIAGNOSTIC_OUTPUT) \
		--summary-output $(PLATEAU_DIAGNOSTIC_SUMMARY) \
		--models-output $(PLATEAU_DIAGNOSTIC_MODELS) \
		--winner-summary-output $(PLATEAU_DIAGNOSTIC_WINNERS) \
		--split $(EVAL_SPLIT) \
		--split-col $(SPLIT_COL) \
		--min-count $(PLATEAU_DIAGNOSTIC_MIN_COUNT) \
		$(PLATEAU_DIAGNOSTIC_ARGS) \
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
