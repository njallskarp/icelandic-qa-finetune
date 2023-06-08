from . import utils
from . import convbert 
from . import icebert
from .utils import ALLOWED_MODELS

def get_model(model_name):
    
    if model_name == Utils.CONVBERT:
        return convbert.load()
    if model_name == Utils.ICEBERT:
        return icebert.load()

    raise ModuleNotFoundError(f"Module for model {model_name} not found")