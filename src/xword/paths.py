from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
SRC = PACKAGE.parent
ROOT = SRC.parent
WORDLIST_PATH = ROOT / "data" / "wordlist.txt"
PUZZLES_DIR = ROOT / "data" / "puzzles"
