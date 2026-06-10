from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "visuallyimpair"

IMAGE_DIR = DATA_DIR / "visual_dataset"
TEXT_DIR = DATA_DIR / "visual_text"

TOKEN_FILE = TEXT_DIR / "visual.token.txt"
TRAIN_IMAGE_FILE = TEXT_DIR / "visual.trainImages.txt"
TEST_IMAGE_FILE = TEXT_DIR / "visual.testImages.txt"

IMAGE_SIZE = (224, 224)
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"

MIN_WORD_FREQ = 1

BATCH_SIZE = 32
NUM_WORKERS = 0
PIN_MEMORY = False
VAL_RATIO = 0.2
RANDOM_SEED = 42

EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0

EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 1
DROPOUT = 0.3