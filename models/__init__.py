from . import utils, convbert, icebert
from .utils import ALLOWED_MODELS

def get_model(model_name):
    
    print(f"Loading {model_name}\n")
    
    if model_name == utils.CONVBERT:
        model = convbert.load()
    if model_name == utils.ICEBERT:
        model = icebert.load()
    
    if not model:
        raise ModuleNotFoundError(f"Module for model {model_name} not found")
    
    print(f"Stuccessfully loaded {model_name}\n")
    return model