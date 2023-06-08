import torch
from transformers import RobertaForQuestionAnswering, AutoTokenizer

def load():
    BERT_MODEL = "Mideind/icebert"
    
    device = torch.device('cuda:0' if torch.cuda.is_available()
                      else 'cpu')
    model = RobertaForQuestionAnswering.from_pretrained(BERT_MODEL).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    
    return model, tokenizer