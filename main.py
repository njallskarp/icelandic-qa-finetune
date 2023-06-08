import argparse
from datasets import ALLOWED_DATASETS, get_data
from models import ALLOWED_MODELS, get_model
import config

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
parser.add_argument('--model_out_file', type=str,  help='Output path for model', required=True)
parser.add_argument('--seed', type=int,  help='Seed for setup', required=False, default = DEFAULT_SEED)

def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def main():
    
    set_seed(DEFAULT_SEED)
    
    args = parser.parse_args()
    
    model, tokenizer = get_model(args.model_name)
    
    train_loader, test_loader = get_data(args.datasets, model, tokenizer, args.batch_size)
    
    run_training(train_loader, test_loader, model, tokenizer, args.epochs, args.lr)
    
    


if __name__ == "__main__":
    main()