# [S:TEST v1] eval_public_script pass
import json
import os
import shutil
import subprocess
from pathlib import Path


def test_eval_public_script_runs(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    data_file = repo_root / "data/arc-agi_evaluation_challenges.json"
    backup = tmp_path / "arc-agi_evaluation_challenges.json.bak"
    shutil.copy(data_file, backup)
    try:
        with open(data_file) as f:
            all_data = json.load(f)
        first_id = next(iter(all_data))
        minimal = {first_id: all_data[first_id]}
        with open(data_file, "w") as f:
            json.dump(minimal, f)
        env = os.environ.copy()
        env["BATCH"] = "1"
        env["OUT"] = "submission/full_submission.json"
        subprocess.run(["bash", str(repo_root / "scripts/eval_public.sh")], cwd=repo_root, check=True, env=env)
        out_file = repo_root / env["OUT"]
        assert out_file.exists()
        with open(out_file) as f:
            sub = json.load(f)
        assert len(sub) == 1
    finally:
        shutil.move(str(backup), data_file)
        site = repo_root / "sitecustomize.py"
        if site.exists():
            site.unlink()
        out_file = repo_root / "submission/full_submission.json"
        if out_file.exists():
            out_file.unlink()
