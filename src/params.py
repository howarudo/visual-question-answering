import os
from dotenv import load_dotenv

load_dotenv()

### TOKENS ###
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")


### LOCAL DATA PATHS ###
INPUT_PATH = 'data'
TRAIN_PATH = INPUT_PATH + '/train'
VALIDATION_PATH = INPUT_PATH + '/valid'
ANNOTATIONS_TRAIN_PATH = INPUT_PATH + '/train.json'
ANNOTATIONS_VAL_PATH = INPUT_PATH + '/valid.json'
OUTPUT_PATH = 'submission'


### HUGGINGFACE REPO IDs ###
MODEL_REPO_ID = "google/paligemma-3b-pt-224"
FINETUNED_MODEL_ID = "howarudo/paligemma-3b-pt-224-vqa-sub-ft"
EVAL_REPO_ID = "howarudo/paligemma-3b-pt-224-vqa-continue-ft-0"


### CONSTANTS ###
MAX_LENGTH = 512
PROMPT = "Answer: "
TEST_BATCH_SIZE = 8
