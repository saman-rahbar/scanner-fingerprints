# Shortcuts for the common tasks. Each target is one or two plain commands, so
# you can also run them by hand.

PY ?= python
SANDBOX := confound_audit combat_baseline scanner_dominance scanner_dominance_ci \
           intrinsic_equivalence population_adjust raw_voxel_baseline multiseed_intrinsic \
           arch_encoders global_readout segdice_intervention segdice_wherebites

.PHONY: install test figures control hero

install:
	$(PY) -m pip install -r requirements.txt

test:  ## synthetic self-tests: no data, no checkpoints, no GPU
	@for s in $(SANDBOX); do echo "== $$s"; $(PY) src/$$s.py --sandbox > /dev/null || exit 1; done
	@echo "all $(words $(SANDBOX)) self-tests passed"

figures:  ## the paper's figures, from the results in results/
	$(PY) src/make_figures.py
	$(PY) src/make_figures_v2.py

control:  ## search for a pretraining advantage, 50 splits against one split
	$(PY) src/positive_control.py results/abide1_dominance_50splits_1p0mm.json
	$(PY) src/positive_control.py results/abide1_dominance_single_split.json

hero:  ## the animated header in assets/
	$(PY) tools/make_hero.py
