import torch
from transformers import ConvBertForQuestionAnswering, AutoTokenizer

def load():
    BERT_MODEL = "jonfd/convbert-base-igc-is"
    
    device = torch.device('cuda:0' if torch.cuda.is_available()
                      else 'cpu')
    model = ConvBertForQuestionAnswering.from_pretrained(BERT_MODEL).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    
    print(f"\n\tConvbert is loaded on {device}")
    
    return model, tokenizer