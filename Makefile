PY ?= python3
OUT ?= submission/full_submission.json
BATCH ?= 50

.PHONY: deps train submit eval_public

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
