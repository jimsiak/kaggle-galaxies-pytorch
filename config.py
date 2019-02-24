'''
Global configuration parameters.
'''

### The path where the dataset has been downloaded
DS_DIR ="/path/to/kaggle-data/"
TRAIN_DIR = DS_DIR + "/images_training_rev1/"
TRAIN_CSV = DS_DIR + "/training_solutions_rev1.csv"
TEST_DIR = DS_DIR + "/images_test_rev1/"
TEST_CSV = DS_DIR + "/all_ones_benchmark.csv"

MODEL = "resnet50" # "sander_dieleman" or "alexnet" or "vgg16" or "resnet50"

SAVE_MODEL = True
MODEL_FILENAME = "model-" + MODEL + ".pt"

OUTPUT_CSV = "output-" + MODEL + ".csv"

NUM_EPOCHS = 5
BATCH_SIZE = 256
DEVICE = "cuda" # "cuda" or "cpu"
NR_DEVICES = 2

### For the split in train and validation datasets
RANDOM_SEED = 42
SHUFFLE_DS = True
VALIDATION_SPLIT = 0.05

'''
Fixed variables.
No need to change anything from here on.
'''

CSV_HEADER = [ "GalaxyID", "Class1.1", "Class1.2", "Class1.3", "Class2.1",
               "Class2.2", "Class3.1", "Class3.2", "Class4.1", "Class4.2",
               "Class5.1", "Class5.2", "Class5.3", "Class5.4", "Class6.1",
               "Class6.2", "Class7.1", "Class7.2", "Class7.3", "Class8.1",
               "Class8.2", "Class8.3", "Class8.4", "Class8.5", "Class8.6",
               "Class8.7", "Class9.1", "Class9.2", "Class9.3", "Class10.1",
               "Class10.2", "Class10.3", "Class11.1", "Class11.2", "Class11.3",
               "Class11.4","Class11.5", "Class11.6"]
