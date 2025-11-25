
# Dataset parameters
BASE_MODEL_ID = "microsoft/phi-2"
DATASET_ID = "garage-bAInd/Open-Platypus"


DATASET_OUTPUT_DIR = "data/platypus"
CHECKPOINT_DIR = "checkpoints"



# Training parameters
TRAIN_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
MAX_STEPS = 100
MAX_LENGTH = 1024
LEARNING_RATE = 2e-5
EVAL_STEPS = 10
SAVE_STEPS = 25
LOGGING_STEPS=10
