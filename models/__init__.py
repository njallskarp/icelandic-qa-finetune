from .utils import ALLOWED_MODELS
from . import conv_bert 

def get_model(model_name):
    
    if model_name == "jonfd/convbert-base-igc-is":
        return ConvBert.load()

    raise ModuleNotFoundError(f"Module for model {model_name} not found")