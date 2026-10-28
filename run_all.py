from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parent
PROJECTS = [
    ROOT / "projects" / "bank-marketing" / "src" / "train.py",
    ROOT / "projects" / "adult-income" / "src" / "train.py",
    ROOT / "projects" / "wine-quality" / "src" / "train.py",
]


def main() -> None:
    for script in PROJECTS:
        print(f"\n=== Running {script.parents[1].name} ===", flush=True)
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
