PY ?= python3
OUT ?= submission/full_submission.json
BATCH ?= 50

.PHONY: deps train submit eval_public eval_agentic eval_genomic eval_ensemble

deps:
$(PY) -m pip install -r requirements.txt

train:
$(PY) -u tools/build_memory.py --train_json data/arc-agi_training_challenges.json
$(PY) -u tools/train_guidance_on_arc.py \
--train-challenges data/arc-agi_training_challenges.json \
--train-solutions  data/arc-agi_training_solutions.json \
--out neural_guidance_model.json

submit:
$(PY) -u arc_submit.py --out $(OUT)

eval_public:
	BATCH=$(BATCH) OUT=$(OUT) bash scripts/eval_public.sh

# Evaluate public ARC subset using the agentic solver
eval_agentic:
	SOLVER=agentic OUT=submission/agentic_submission.json BATCH=$(BATCH) bash scripts/eval_with_solver.sh

# Evaluate public ARC subset using the genomic solver
eval_genomic:
	SOLVER=genomic OUT=submission/genomic_submission.json BATCH=$(BATCH) bash scripts/eval_with_solver.sh

# Evaluate using ensemble of new solvers
eval_ensemble:
	SOLVER=ensemble_new OUT=submission/ensemble_submission.json BATCH=$(BATCH) bash scripts/eval_with_solver.sh
