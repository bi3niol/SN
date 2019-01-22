COMITET_MODELS = [ "./Data/model_2_weights_1.h5", "./Data/model_3_weights_1.h5", "./Data/model_4_weights_1.h5", "./Data/model_5_weights_1.h5"]
COMITET_PREDICT = False
CREATE_THRESHOLD_TEST_FOR_COMITET = False
COMITET_THRESHOLDS = [0.3,0.4,0.5,0.6]
THRESHOLD_TEST_FOR_COMITET_OUT_FILE = "./Data/Threshold_Test.csv"

TEST_MODELS_IN_COMITET_OUT_FILE = "./Data/Models_Test.csv"
TEST_MODELS_IN_COMITET = False

CREATE_COMITET = CREATE_THRESHOLD_TEST_FOR_COMITET or COMITET_PREDICT or TEST_MODELS_IN_COMITET



IMAGES_DIR = "./Data/images"
TEST_DIR = "./Data/test"
MASKS_DIR = "./Data/labels"
MODEL_CHECKPOINT = "./Data/model_6_weights_1.h5"

MODEL_LOCATION = "./Data/model_2_weights_1.h5"
HISTORY_PATH = "./Data/history6_m1.json"
EPOCH_PATH = "./Data/epochs6_m1.json"

IMG_WIDTH = 400
IMG_HEIGHT = 400
BATCH_SIZE = 2

THRESHOLD = 0.4
PREDICT_DATA_THRESHOLD = 0.4
PRINT_MODEL = False
PLOT_MODEL = False
TRAIN_MODEL = False
TEST_MODEL = False
TEST_IMAGE_IDS = ["23", "24", "25", "26", "27"]
CREATE_MODEL = PRINT_MODEL or PLOT_MODEL or TRAIN_MODEL or TEST_MODEL
