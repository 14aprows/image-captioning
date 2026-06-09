from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "visuallyimpair"

IMAGE_DIR = DATA_DIR / "visual_dataset"
TEXT_DIR = DATA_DIR / "visual_text"

TOKEN_FILE = TEXT_DIR / "visual.token.txt"
TRAIN_IMAGE_FILE = TEXT_DIR / "visual.trainImages.txt"
TEST_IMAGE_FILE = TEXT_DIR / "visual.testImages.txt"