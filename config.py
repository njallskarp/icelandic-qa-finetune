from dotenv import dotenv_values

config = dotenv_values(".env") 

WANDB_ENTITY = config['WANDB_ENTITY']
LABELSTUDIO_TOKEN = config['LABELSTUDIO_TOKEN']
DEFAULT_SEED = config['DEFAULT_SEED']
