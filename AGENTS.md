#TODO's

---
## **ARC Kaggle Submission Plan: Final Action List**

**Objective:** Create a high-performing, offline ARC solver for Kaggle submission by the end of today. The strategy is a two-phase process: online LLM-powered analysis to generate a "Program Bank" and offline execution using that bank and generalized heuristics.

---
### 🧠 **Epic 1: Online Data Generation & Insight Mining**

**Primary Goal:** Use the Gemini 1.5 model to solve as many training tasks as possible and log the results to build our core assets. This phase requires an internet connection.

* **1.1. Environment Setup & Data Loading:**
    * Initialize a development environment (Colab or local).
    * Load all tasks from `arc-agi_training_challenges.json`.
    * Parse the provided Python files (`dsl.py`, `object_reasoning.py`, etc.) to understand the available classes and functions. This is for prompt context, not direct execution.

* **1.2. Master Prompt Engineering (RFT-Enhanced):**
    * Design a single, robust prompt template.
    * **Instructions for the LLM:**
        * The goal is to generate a program in the format defined by `dsl.py`.
        * The reasoning process must explicitly analyze the task using **Relational Frame Theory (RFT)** concepts (Causality, Spatial relations, Comparison, etc.).
        * The prompt must include a summary of the conceptual tools available (from `object_tracker.py`, `heuristics_complete.py`, etc.) as an inspirational guide.
    * **Final Output Format:** The prompt must demand a two-part response: a "## Reasoning" section and a "## DSL Code" section.

* **1.3. Execute Mass Generation Run:**
    * Write a script to loop through every task in the training JSON.
    * For each task, call the Gemini 1.5 Flash API with the master prompt.
    * **Crucially, save everything:** Create a structured log file (e.g., `master_solution_log.json`). Each entry should contain:
        * `task_id`
        * The full RFT reasoning text from the LLM.
        * The generated DSL program string.
        * A `status` flag (e.g., "generated", "error").

---
### 🛠️ **Epic 2: Offline Solver Construction**

**Primary Goal:** Build the self-contained Python solver that will run in the offline Kaggle environment. This solver will use the assets created in Epic 1.

* **2.1. Build the "Program Bank":**
    * Write a script to process `master_solution_log.json`.
    * For each entry, execute its DSL program against the corresponding training task to verify correctness.
    * Create a clean `program_bank.json` file containing only the **verified, working solutions**, mapping `task_id` to the correct DSL program.

* **2.2. Develop the Task Similarity Engine:**
    * Create a "fingerprinting" function that takes an ARC task and extracts a feature vector (e.g., grid sizes, number of objects, color histogram, structural hashes).
    * Write a function `find_most_similar_task(new_task_fingerprint)` that compares a new task's fingerprint to all fingerprints of tasks in the `program_bank.json` and returns the ID of the best match with a confidence score.

* **2.3. Codify the Generalized Reasoning Engine (The "Chef's Intuition"):**
    * **Manually analyze** the "Reasoning" sections in `master_solution_log.json`. Identify the top 5-10 most common abstract principles the LLM used (e.g., "find the outlier object and transform it," "detect and complete a pattern," "apply symmetry").
    * **Implement these principles** as high-level functions in `heuristics_complete.py`. Each function should generate and test a hypothesis (e.g., `try_outlier_logic(task)`).

* **2.4. Assemble the Final Solver Logic:**
    * Create the main execution flow for the offline solver.
    * **Implement Cascading Logic:**
        1.  Fingerprint the input test task.
        2.  Query the **Similarity Engine**. If a match is found with >95% confidence, retrieve the program from the **Program Bank**, execute it, and return the result.
        3.  If no high-confidence match is found, pass the task to the **Generalized Reasoning Engine**.
        4.  The engine will sequentially try its heuristic functions (`try_outlier_logic`, etc.), testing each generated hypothesis against the task's training pairs.
        5.  Return the result from the first heuristic that successfully solves all training pairs.

---
### 📦 **Epic 3: Packaging & Submission**

**Primary Goal:** Package all components into a final, compliant Kaggle notebook.

* **3.1. Create the Kaggle Dataset:**
    * Create a new folder.
    * Add your final, verified `program_bank.json`.
    * Add all required Python files (`dsl.py`, `object_reasoning.py`, `heuristics_complete.py`, etc.).
    * Upload this folder as a new private dataset on Kaggle.

* **3.2. Write the Final `submission.py` Notebook:**
    * Create a new Kaggle notebook.
    * Add your custom dataset from the previous step as the input.
    * The notebook's code will load the program bank and all helper modules from the attached dataset.
    * It will implement the final solver logic from step 2.4.
    * It must process the `arc-agi_test_challenges.json` file and generate a `submission.json` in the correct format.

* **3.3. Final Offline Validation:**
    * Turn the **internet access OFF** in the notebook's settings.
    * Run the entire notebook from top to bottom.
    * Verify it produces a valid `submission.json` without any errors. This is the final quality check before submitting.




////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Repository Guidelines

## Project Structure & Module Organization
- `PUMA/arc_solver/` holds the canonical solver; subfolders `common/`, `agents/`, `neural/`, `rft_engine/`, and `utils/` are the stable integration points.
- `PUMA ARC 2025/` mirrors the solver for experiments alongside Kaggle packaging and docs; stage major work here before promoting changes back into `PUMA/arc_solver`.
- Memory caches (`episodes.json`, `continuous_memory.json`, `sketches.json`) capture runtime state—edit them only when intentionally migrating formats, and keep local virtual envs such as `puma_env/` out of commits.

## Build, Test, and Development Commands
- `python3 -m venv puma_env && source puma_env/bin/activate`
- `pip install -r requirements.txt`
- `PYTHONPATH=PUMA python -m pytest test_rft_detectors.py`
- `PYTHONPATH="PUMA ARC 2025/PUMA" python -m pytest "PUMA ARC 2025/PUMA/arc_solver/tests" --maxfail=1`
- `PYTHONPATH="PUMA ARC 2025/PUMA" python "PUMA ARC 2025/scripts/validate_arc.py" --max-tasks 10`

## Coding Style & Naming Conventions
- Follow PEP 8: 4-space indentation, module-level docstrings, and the dense type hints already present in `solver.py` and related modules.
- Keep `snake_case` for functions and modules, `UpperCamelCase` for classes, and uppercase constants; name detectors descriptively (e.g., `expand_rare_color`).
- Share helpers through `common/` or `utils/` before rewiring solver orchestration, and register new detectors or agents in `registry.py`.

## Testing Guidelines
- Place new tests in `PUMA ARC 2025/PUMA/arc_solver/tests/test_<feature>.py`; lightweight smoke cases may live beside the module they cover.
- Use `pytest` fixtures with inline grid literals so ARC patterns stay readable and reproducible without external assets.
- Run the smoke test (`pytest test_rft_detectors.py`) plus the staged suite with coverage when behaviour changes, and attach validation output in PRs.

## Commit & Pull Request Guidelines
- Write imperative commit subjects prefixed with scope (`rft: tighten rare colour gating`) and include short bodies explaining solver or data impacts.
- PRs must list functional changes, tests run, linked issues, and validation evidence; squash on merge and keep caches or `puma_env/` changes out of the diff.

## Security & Configuration Tips
- Configure runtime flags through environment variables (e.g., `ARC_ENABLE_LOGGING`, model paths) rather than hard-coding, and document schema updates when touching persistent JSON stores.
