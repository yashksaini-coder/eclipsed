"""Inline the generator into each task so it is one paste-able Kaggle file."""
import pathlib

root = pathlib.Path(__file__).parent
gen = (root / "eclipsed" / "gen.py").read_text().split("\n# --------------------------------------------------------------------------\n\nFAMILIES")[0]
(root / "dist").mkdir(exist_ok=True)
for task in sorted((root / "tasks").glob("*.py")):
    out = root / "dist" / task.name
    out.write_text(
        "# Built by build.py from eclipsed/gen.py + tasks/" + task.name + ". Edit the sources, not this file.\n\n"
        "# %%\n# Item generator (standard library only)\n" + gen.rstrip() + "\n\n\n" + task.read_text()
    )
    print("built", out.relative_to(root))
