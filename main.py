import argparse
from datasets import ALLOWED_DATASETS
from models import ALLOWED_MODELS

ALLOWED_MODELS = ['mideind/Icebert']

def check_model_name(value):
    if value not in ALLOWED_MODELS:
        raise argparse.ArgumentTypeError(f"Invalid model name: {value}. Allowed values are {ALLOWED_MODELS}")
    return value

def check_dataset_names(value):
    if value not in ALLOWED_DATASETS:
        raise argparse.ArgumentTypeError(f"Invalid dataset name: {value}. Allowed values are {ALLOWED_DATASETS}")
    return value

parser = argparse.ArgumentParser(description='BERT model training script')

parser.add_argument('--model_name', type=check_model_name, required=True,
                    help='Name of the BERT model to use. Allowed values: ' + ', '.join(ALLOWED_MODELS))
parser.add_argument('--datasets', type=check_dataset_names, required=True, nargs='+',
                    help='List of dataset names to use. Allowed values: ' + ', '.join(ALLOWED_DATASETS))
parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for the optimizer')
parser.add_argument('--epochs', type=int, default=3, help='Number of epochs for training')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')



def main():
    args = parser.parse_args()
    print(args) 


if __name__ == "__main__":
    main()