from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT / "src"))
