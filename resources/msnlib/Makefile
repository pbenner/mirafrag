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
PREDICTIONS ?= $(RESOURCE_DIR)/predictions/msnlib_test_predictions.csv
DISK_CACHE_DIR ?= $(RESOURCE_DIR)/data/cache/mirafrag_features
MODEL_ABS := $(abspath $(MODEL))
PREDICTIONS_ABS := $(abspath $(PREDICTIONS))
DISK_CACHE_DIR_ABS := $(abspath $(DISK_CACHE_DIR))

.PHONY: help prepare train eval

help:
	@printf '%s\n' \
		'MSnLib MiraFrag targets:' \
		'  make prepare          Convert FIORA MSnLib library.csv to MiraFrag TSV' \
		'  make train            Train through the shared MassSpecGym runner' \
		'  make eval             Evaluate through the shared MassSpecGym runner' \
		'' \
		'Common variables:' \
		'  SOURCE=$(SOURCE)' \
		'  INPUT=$(INPUT)' \
		'  INSTRUMENT_SOURCE=$(INSTRUMENT_SOURCE)' \
		'  MODEL=$(MODEL)' \
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

eval:
	$(MAKE) -C $(MASSPECGYM_RESOURCE_DIR) eval \
		INPUT=$(INPUT) \
		MODEL=$(MODEL_ABS) \
		PREDICTIONS=$(PREDICTIONS_ABS) \
		DISK_CACHE_DIR=$(DISK_CACHE_DIR_ABS) \
		ENCODER=$(ENCODER) \
		CHECKPOINT_ENCODER=$(CHECKPOINT_ENCODER)
