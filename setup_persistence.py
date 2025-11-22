"""
Setup Persistence Database

Initialize atomspace persistence directory.
"""

from pathlib import Path
import json


def setup_persistence(base_path: Path = Path("./atomspace-db")):
    """
    Set up persistence directory structure.

    Args:
        base_path: Base path for atomspace database
    """
    base_path = Path(base_path)

    # Create directories
    (base_path / "default").mkdir(parents=True, exist_ok=True)
    (base_path / "default" / "snapshots").mkdir(exist_ok=True)

    # Create initial empty atoms file
    atoms_file = base_path / "default" / "atoms.json"
    if not atoms_file.exists():
        with open(atoms_file, 'w') as f:
            json.dump({}, f)

    # Create initial empty links file
    links_file = base_path / "default" / "links.json"
    if not links_file.exists():
        with open(links_file, 'w') as f:
            json.dump([], f)

    # Create config
    config_file = base_path / "config.json"
    config = {
        "version": "1.0",
        "default_db": "default",
        "auto_save_interval": 60,
        "snapshot_interval": 3600
    }

    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Persistence setup complete at: {base_path}")
    print(f"   Default database: {base_path / 'default'}")


if __name__ == "__main__":
    setup_persistence()
