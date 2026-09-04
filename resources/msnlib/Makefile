UV ?= uv
RESOURCE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(RESOURCE_DIR)/../..)
MASSPECGYM_RESOURCE_DIR := $(ROOT)/resources/massspecgym

ENCODER ?= unimol
CHECKPOINT_ENCODER ?= $(if $(filter unimol,$(strip $(ENCODER))),unimol,all)
DEV_PYTHON ?= $(UV) run --project $(ROOT) --extra dev python
PYTHONPATH := $(ROOT)/src

SOURCE ?= $(abspath $(ROOT)/../fiora/resources/data/msnlib/library.csv)
INPUT ?= $(RESOURCE_DIR)/data/MSnLib_mirafrag.tsv
INSTRUMENT_SOURCE ?= fragmentation
MODEL ?= $(RESOURCE_DIR)/checkpoints/mirafrag_msnlib.pt
ROBUST_MODEL ?= $(RESOURCE_DIR)/checkpoints/mirafrag_msnlib_unimol_robust.pt
ACTION_GEOMETRY_MODEL ?= $(RESOURCE_DIR)/checkpoints/mirafrag_msnlib_unimol_action_geometry.pt
PREDICTIONS ?= $(RESOURCE_DIR)/predictions/msnlib_test_predictions.csv
DISK_CACHE_DIR ?= $(RESOURCE_DIR)/data/cache/mirafrag_features
MODEL_ABS := $(abspath $(MODEL))
PREDICTIONS_ABS := $(abspath $(PREDICTIONS))
DISK_CACHE_DIR_ABS := $(abspath $(DISK_CACHE_DIR))

.PHONY: help prepare train train-action-geometry eval

help:
	@printf '%s\n' \
		'MSnLib MiraFrag targets:' \
		'  make prepare          Convert FIORA MSnLib library.csv to MiraFrag TSV' \
		'  make train            Train through the shared MassSpecGym runner' \
		'  make train-action-geometry  Reproduce the current MSnLib action-geometry fine-tune' \
		'  make eval             Evaluate through the shared MassSpecGym runner' \
		'' \
		'Common variables:' \
		'  SOURCE=$(SOURCE)' \
		'  INPUT=$(INPUT)' \
		'  INSTRUMENT_SOURCE=$(INSTRUMENT_SOURCE)' \
		'  MODEL=$(MODEL)' \
		'  ROBUST_MODEL=$(ROBUST_MODEL)' \
		'  ACTION_GEOMETRY_MODEL=$(ACTION_GEOMETRY_MODEL)' \
		'  PREDICTIONS=$(PREDICTIONS)' \
		'  DISK_CACHE_DIR=$(DISK_CACHE_DIR)' \
		'  ENCODER=$(ENCODER)'

prepare:
	mkdir -p $(dir $(INPUT))
	PYTHONPATH=$(PYTHONPATH) $(DEV_PYTHON) -m mirafrag.cli.prepare_msnlib \
		--input $(SOURCE) \
		--output $(INPUT) \
		--instrument-source $(INSTRUMENT_SOURCE)

train:
	$(MAKE) -C $(MASSPECGYM_RESOURCE_DIR) train \
		INPUT=$(INPUT) \
		MODEL=$(MODEL_ABS) \
		DISK_CACHE_DIR=$(DISK_CACHE_DIR_ABS) \
		ENCODER=$(ENCODER) \
		CHECKPOINT_ENCODER=$(CHECKPOINT_ENCODER)

train-action-geometry:
	$(MAKE) train \
		MODEL=$(ACTION_GEOMETRY_MODEL) \
		INIT_MODEL=$(ROBUST_MODEL) \
		ENCODER=unimol \
		CHECKPOINT_ENCODER=unimol \
		LOSS=sqrt_cosine \
		CHECKPOINT_METRIC=val_cosine \
		LR=1e-5 \
		ENCODER_LR=3e-6 \
		WEIGHT_DECAY=0 \
		HEAD_WEIGHT_DECAY=0 \
		TARGET_POWER=1.0 \
		ENTROPY_WEIGHT=0 \
		DROPOUT=0.1 \
		SCHEDULER=exponential \
		EXPONENTIAL_GAMMA=0.98 \
		TRAIN_PREFILL_CACHE=0 \
		FRAGMENT_PATH_LAYERS=3 \
		FRAGMENT_ACTION_PRIMARY_LAYERS=2 \
		BOND_BREAK_GEOMETRY_FEATURES=1

eval:
	$(MAKE) -C $(MASSPECGYM_RESOURCE_DIR) eval \
		INPUT=$(INPUT) \
		MODEL=$(MODEL_ABS) \
		PREDICTIONS=$(PREDICTIONS_ABS) \
		DISK_CACHE_DIR=$(DISK_CACHE_DIR_ABS) \
		ENCODER=$(ENCODER) \
		CHECKPOINT_ENCODER=$(CHECKPOINT_ENCODER)
