# config.py
import transformers

# this is the maximum number of tokens in the sentence
MAX_LEN = 512

# batch size is small because model is huge
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4

# let's train for a maximum of 10 epoches
EPOCHS = 10

# define path to BERT model files
BERT_PATH = "../input/bert-base-uncased/"

# this is where you want to save the model
MODEL_PATH = "model.bin"

# training file
TRAINING_FILE = "../input/IMDB Dataset.csv"

# define the tokenizer
# we use tokenizer and model
# from huggingface's transformers
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)

