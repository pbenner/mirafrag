UV ?= uv
RESOURCE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(RESOURCE_DIR)/../..)
ENCODER ?= aimnet
CHECKPOINT_ENCODER ?= all
PYTHON ?= $(UV) run --project $(ROOT) --extra $(ENCODER) python
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
PREDICT_OUTPUT ?= $(RESOURCE_DIR)/predictions/mirafrag_predictions.csv
DISK_CACHE_DIR ?= $(RESOURCE_DIR)/data/cache/mirafrag_features
CACHE_SOURCE_MODEL ?=
CACHE_SPLITS ?= train val test
CACHE_NUM_WORKERS ?= $(shell nproc)
CACHE_CHUNK_SIZE ?= 1

DATASET_REPO ?= roman-bushuiev/MassSpecGym
DATASET_FILE ?= data/MassSpecGym.tsv
DATASET_REVISION ?= main
DATASET_LOCAL_DIR ?= $(RESOURCE_DIR)

FOUNDATION_SOURCE ?= off
FOUNDATION_MODEL ?= medium
FOUNDATION_PATH ?=
AIMNET_MODEL ?= aimnet2
AIMNET_PATH ?=
DEVICE ?= auto

EPOCHS ?= 200
BATCH_SIZE ?= 48
LR ?= 1e-3
WEIGHT_DECAY ?= 0.0
SCHEDULER ?= exponential
SCHEDULER_INTERVAL ?= epoch
MIN_LR_RATIO ?= 0.0001
EXPONENTIAL_GAMMA ?= 0.85
PLATEAU_FACTOR ?= 0.5
PLATEAU_PATIENCE ?= 1
MZ_MAX ?= 1005
BIN_WIDTH ?= 0.01
LOSS ?= kl
KL_WEIGHT ?= 0.7
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
MAX_FRAGMENT_BROKEN_BONDS ?= 6
MAX_FRAGMENTS ?= 2048
MAX_FRAGMENT_EDGES ?= 8192
FRAGMENT_GNN_LAYERS ?= 2
INCLUDE_FRAGMENT_ISOTOPES ?= 1
FRAGMENT_ISOTOPE_THRESHOLD ?= 0.001
MAX_FRAGMENT_ISOTOPE_PEAKS ?= 1
FINE_TUNE_STRATEGY ?= full
OVERRIDE_FRAGMENT_CONFIG ?= 0

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
else
FOUNDATION_ARGS += --foundation-source $(FOUNDATION_SOURCE) --foundation-model $(FOUNDATION_MODEL)
ifneq ($(strip $(FOUNDATION_PATH)),)
FOUNDATION_ARGS += --foundation-path $(FOUNDATION_PATH)
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

COMMON_ARGS = --device $(DEVICE) --mz-max $(MZ_MAX) --bin-width $(BIN_WIDTH) --batch-size $(BATCH_SIZE) --num-workers $(NUM_WORKERS) --checkpoint-metric $(CHECKPOINT_METRIC) $(PROGRESS_ARGS)
SAMPLE_DEBUG_ARGS = --slow-sample-seconds $(SLOW_SAMPLE_SECONDS)
ifeq ($(strip $(TRACE_SAMPLES)),1)
SAMPLE_DEBUG_ARGS += --trace-samples
else
SAMPLE_DEBUG_ARGS += --no-trace-samples
endif
LOADER_DEBUG_ARGS = $(SAMPLE_DEBUG_ARGS) --dataloader-timeout $(DATALOADER_TIMEOUT)
ARCH_ARGS = --hidden-dim $(HIDDEN_DIM) --num-layers $(NUM_LAYERS) --dropout $(DROPOUT) --metadata-dim $(METADATA_DIM) --fragment-gnn-layers $(FRAGMENT_GNN_LAYERS)
FRAGMENT_ARGS = --max-fragment-tree-depth $(MAX_FRAGMENT_TREE_DEPTH) --max-fragment-broken-bonds $(MAX_FRAGMENT_BROKEN_BONDS) --max-fragments $(MAX_FRAGMENTS) --max-fragment-edges $(MAX_FRAGMENT_EDGES) $(ISOTOPE_FRAGMENT_ARGS) --fragment-isotope-threshold $(FRAGMENT_ISOTOPE_THRESHOLD) --max-fragment-isotope-peaks $(MAX_FRAGMENT_ISOTOPE_PEAKS)
HEAD_ARGS = $(ARCH_ARGS) $(FRAGMENT_ARGS)
CHECKPOINT_HEAD_ARGS =
ifeq ($(strip $(OVERRIDE_FRAGMENT_CONFIG)),1)
CHECKPOINT_HEAD_ARGS += $(FRAGMENT_ARGS)
endif
TRAIN_HEAD_ARGS = $(HEAD_ARGS)
ifneq ($(strip $(INIT_MODEL)),)
TRAIN_HEAD_ARGS = $(CHECKPOINT_HEAD_ARGS)
endif
CACHE_HEAD_ARGS = $(FRAGMENT_ARGS)
ifeq ($(strip $(CACHE_USES_CHECKPOINT)),1)
CACHE_HEAD_ARGS = $(CHECKPOINT_HEAD_ARGS)
endif
FINE_TUNE_ARGS = --fine-tune-strategy $(FINE_TUNE_STRATEGY)
INIT_ARGS =
ifneq ($(strip $(INIT_MODEL)),)
INIT_ARGS += --init-checkpoint $(INIT_MODEL)
endif
LOSS_ARGS = --loss $(LOSS) --kl-weight $(KL_WEIGHT) --coverage-weight $(COVERAGE_WEIGHT) --target-power $(TARGET_POWER) --entropy-weight $(ENTROPY_WEIGHT)
ifeq ($(strip $(RELATIVE_MASS_TOLERANCE)),1)
LOSS_ARGS += --relative-mass-tolerance
else
LOSS_ARGS += --no-relative-mass-tolerance
endif
LOSS_ARGS += --mass-tolerance $(MASS_TOLERANCE) --mass-tolerance-min-mz $(MASS_TOLERANCE_MIN_MZ)
EVAL_DIAGNOSTIC_ARGS = --mass-tolerance $(MASS_TOLERANCE) --mass-tolerance-min-mz $(MASS_TOLERANCE_MIN_MZ)
ifeq ($(strip $(RELATIVE_MASS_TOLERANCE)),1)
EVAL_DIAGNOSTIC_ARGS += --relative-mass-tolerance
else
EVAL_DIAGNOSTIC_ARGS += --no-relative-mass-tolerance
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
OPTIM_ARGS = --learning-rate $(LR) --weight-decay $(WEIGHT_DECAY) $(SCHEDULER_ARGS)
ROW_LIMIT_ARGS =
ifneq ($(strip $(MAX_ROWS)),)
ROW_LIMIT_ARGS += --max-rows $(MAX_ROWS)
endif

.PHONY: help install install-dev install-data-deps prepare-data prepare-cache train train-finetune train-delta eval predict test lint format clean

help:
	@printf '%s\n' \
		'MiraFrag targets:' \
		'  make install             Sync the default uv environment' \
		'  make install-dev         Sync uv environment with dev tools' \
		'  make prepare-data        Download MassSpecGym TSV from Hugging Face' \
		'  make prepare-cache       Precompute graph/fragment feature cache' \
		'  make train INPUT=...     Train spectrum model using FINE_TUNE_STRATEGY' \
		'  make train-finetune      Same as train, with explicit fine-tune target name' \
		'  make train-delta         Continue DELTA_INIT_MODEL with delta fine-tuning' \
		'  make eval INPUT=...      Evaluate checkpoint on MassSpecGym split' \
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
		'  ENCODER=$(ENCODER)' \
		'  CHECKPOINT_ENCODER=$(CHECKPOINT_ENCODER)' \
		'  FOUNDATION_SOURCE=$(FOUNDATION_SOURCE)' \
		'  FOUNDATION_MODEL=$(FOUNDATION_MODEL)' \
		'  AIMNET_MODEL=$(AIMNET_MODEL)' \
		'  DEVICE=$(DEVICE)' \
		'  EPOCHS=$(EPOCHS)' \
		'  BATCH_SIZE=$(BATCH_SIZE)' \
		'  LR=$(LR)' \
		'  SCHEDULER=$(SCHEDULER)' \
		'  LOSS=$(LOSS)' \
		'  KL_WEIGHT=$(KL_WEIGHT)' \
		'  TARGET_POWER=$(TARGET_POWER)' \
		'  ENTROPY_WEIGHT=$(ENTROPY_WEIGHT)' \
		'  MASS_TOLERANCE=$(MASS_TOLERANCE)' \
		'  RELATIVE_MASS_TOLERANCE=$(RELATIVE_MASS_TOLERANCE)' \
		'  MASS_TOLERANCE_MIN_MZ=$(MASS_TOLERANCE_MIN_MZ)' \
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
		'  MAX_FRAGMENT_TREE_DEPTH=$(MAX_FRAGMENT_TREE_DEPTH)' \
		'  MAX_FRAGMENT_BROKEN_BONDS=$(MAX_FRAGMENT_BROKEN_BONDS)' \
		'  MAX_FRAGMENTS=$(MAX_FRAGMENTS)' \
		'  OVERRIDE_FRAGMENT_CONFIG=$(OVERRIDE_FRAGMENT_CONFIG)' \
		'  FINE_TUNE_STRATEGY=$(FINE_TUNE_STRATEGY)' \
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

prepare-cache:
	mkdir -p $(DISK_CACHE_DIR)
	PYTHONPATH=$(PYTHONPATH) $(CACHE_PYTHON) -m mirafrag.cli.cache \
		--input $(INPUT) \
		--disk-cache-dir $(DISK_CACHE_DIR) \
		$(CACHE_MODEL_ARGS) \
		--device $(DEVICE) --mz-max $(MZ_MAX) --bin-width $(BIN_WIDTH) --chunk-size $(CACHE_CHUNK_SIZE) --num-workers $(CACHE_NUM_WORKERS) $(PROGRESS_ARGS) \
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
		$(LOADER_DEBUG_ARGS) \
		$(TRAIN_HEAD_ARGS) \
		$(CACHE_ARGS) \
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
		$(LOADER_DEBUG_ARGS) \
		$(TRAIN_HEAD_ARGS) \
		$(CACHE_ARGS) \
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

eval:
	mkdir -p $(dir $(PREDICTIONS))
	PYTHONPATH=$(PYTHONPATH) $(CHECKPOINT_PYTHON) -m mirafrag.cli.eval \
		--input $(INPUT) \
		--model $(MODEL) \
		--output $(PREDICTIONS) \
		$(COMMON_ARGS) \
		$(CACHE_ARGS) \
		$(EVAL_DIAGNOSTIC_ARGS) \
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
		$(CACHE_ARGS)

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) -q $(ROOT)/tests

lint:
	$(RUFF) check $(ROOT)

format:
	$(RUFF) format $(ROOT)

clean:
	rm -rf $(ROOT)/.pytest_cache $(ROOT)/.ruff_cache $(ROOT)/src/mirafrag/__pycache__ $(ROOT)/src/mirafrag/cli/__pycache__ $(ROOT)/tests/__pycache__
