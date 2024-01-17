"""
config settings:
"""
### TF-BRT parameters
BATCH_SIZE = 72 # Training batch size
TEST_BATCH_SIZE = 1 # Test batch size
EPOCHS = 300 # Traning epoch
SAVE_INTERVAL = 20 # Save model every 20 epochs
STEP_SIZE = 200 * 2 # Step size for moving forward the window (For training) (ronin: 200 * 5; ridi: 200 * 2; neurit: 200 * 2)
TEST_STEP_SIZE = 200 # Step size for moving forward the window (For testing) (ronin: 400; ridi: 200; neurit: 200)
WINDOW_SIZE = 200 # Window size for training and testing (ronin: 400; ridi: 200; oxiod: 200)
SLIDING_SIZE = 200 * 15 # Sliding size for training and testing (ronin: 400 * 20; ridi: 200 * 15; neurit: 200 * 15)
TEST_SLIDING_SIZE = 200 # Sliding size for testing (ronin: 400; ridi: 200; neurit: 200)
PROJECTION_WIDTH = 1 # Projection width
SAMPLING_RATE = 200 # Sampling rate (ronin: 200; ridi: 200; oxiod: 100)
USE_MAGNETOMETER = True # Use magnetometer or not
USE_ANGMENTATION = True # Use angmentation or not
ENABLE_COMPLEMENTARY = False # Enable complementary filter
INPUT_CHANNEL = 9 # Input feature dimension (Gryo + Acce + Magn)
OUTPUT_CHANNEL = 2 # Output dimension (2D deviation vector)
DROPOUT = 0.1 # Dropout probability
LEARNING_RATE = 0.0003 # Learning rate (ronin: 0.0002, ridi: 0.0002, robot: 0.00003)
NUM_WORKERS = 8
### ------------------ ###

### Data preprocessing parameters
FEATURE_SIGMA = 2.0 # Sigma for feature gaussian smoothing
TARGET_SIGMA = 30.0 # Sigma for target gaussian smoothing
### ------------------ ###

### Device for training
DEVICE = "cuda:0" # You can choose GPU or CPU
### ------------------ ###

### Training and testing setting
DATASET = "neurit" # Dataset name
MODEL_TYPE = "tf-brt"
DATA_DIR = './NeurIT Dataset/uniform_data/train_dataset' # Dataset directory for training
VAL_DATA_DIR = './NeurIT Dataset/uniform_data/val_dataset' # Dataset directory for validation
TEST_DIR = './NeurIT Dataset/uniform_data/test_seen' # Dataset directory for testing (test_seen & test_unseen)
OUT_DIR = './prediction_model/neurit/test1' # Output directory for both traning and testing
MODEL_PATH = './prediction_model/' # Model path for testing
### ------------------ ###

def load_config():
    kwargs = {}
    kwargs['batch_size'] = BATCH_SIZE
    kwargs['test_batch_size'] = TEST_BATCH_SIZE
    kwargs['epochs'] = EPOCHS
    kwargs['save_interval'] = SAVE_INTERVAL
    kwargs['step_size'] = STEP_SIZE
    kwargs['test_step_size'] = TEST_STEP_SIZE
    kwargs['window_size'] = WINDOW_SIZE
    kwargs['sliding_size'] = SLIDING_SIZE
    kwargs['test_sliding_size'] = TEST_SLIDING_SIZE
    kwargs['projection_width'] = PROJECTION_WIDTH
    kwargs['sampling_rate'] = SAMPLING_RATE
    kwargs['use_magnetometer'] = USE_MAGNETOMETER
    kwargs['use_angmentation'] = USE_ANGMENTATION
    kwargs['enable_complementary'] = ENABLE_COMPLEMENTARY
    kwargs['input_channel'] = INPUT_CHANNEL
    kwargs['output_channel'] = OUTPUT_CHANNEL
    kwargs['dropout'] = DROPOUT
    kwargs['learning_rate'] = LEARNING_RATE
    kwargs['num_workers'] = NUM_WORKERS

    kwargs['feature_sigma'] = FEATURE_SIGMA
    kwargs['target_sigma'] = TARGET_SIGMA

    kwargs['device'] = DEVICE

    kwargs['dataset'] = DATASET
    kwargs['model_type'] = MODEL_TYPE
    kwargs['data_dir'] = DATA_DIR
    kwargs['val_data_dir'] = VAL_DATA_DIR
    kwargs['test_dir'] = TEST_DIR
    kwargs['out_dir'] = OUT_DIR
    kwargs['model_path'] = MODEL_PATH

    return kwargs
