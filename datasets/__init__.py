from . import ruquad_labeling
from .config import ALLOWED_DATASETS
from . import config
from transformers import AutoTokenizer,AdamW,BertForQuestionAnswering
from torch.utils.data import DataLoader, Dataset
import torch 
import random

def __map_name_to_module(name):
    
    if name not in ALLOWED_DATASETS:
        raise ModuleNotFoundError(f"The dataset requested '{name}' not found")
    if name == config.RUQUAD_LABELING:
        return ruquad_labeling
    else:
        raise ModuleNotFoundError(f"The dataset requested '{name}' not found")


def __correct_span_errors(answers, texts):
    """
    Code reused from open source colab notebook:
    https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset/blob/main/Fine_Tuning_Bert.ipynb
    """
    for answer, text in zip(answers, texts):
        real_answer = answer['text']
        start_idx = answer['answer_start']
        # Get the real end index
        end_idx = start_idx + len(real_answer)

        # Deal with the problem of 1 or 2 more characters 
        if text[start_idx:end_idx] == real_answer:
            answer['answer_end'] = end_idx
        # When the real answer is more by one character
        elif text[start_idx-1:end_idx-1] == real_answer:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1  
        # When the real answer is more by two characters  
        elif text[start_idx-2:end_idx-2] == real_answer:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2    
            
def __add_token_positions(encodings, tokenizer, answers):
    """
    Code reused from open source colab notebook:
    https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset/blob/main/Fine_Tuning_Bert.ipynb
    """
    
    start_pos, end_pos = [], []
    
    for i, answer in enumerate(answers):
        start = encodings.char_to_token(i, answer['answer_start'])
        end   = encodings.char_to_token(i, answer['answer_end'])
        
        if start is None:
            start = tokenizer.model_max_length
        if end is None:
            end = encodings.char_to_token(i, answer['answer_end'] - 1)
        if end  is None:
            end = tokenizer.model_max_length
            
        start_pos.append(start)
        end_pos.append(end)
        
    encodings.update({'start_positions': start_pos, 'end_positions': end_pos})
    
class SquadDataset(torch.utils.data.Dataset):
    """
    Code reused from open source colab notebook:
    https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset/blob/main/Fine_Tuning_Bert.ipynb
    """
    def __init__(self, encodings, is_train = True):
        self.encodings = encodings
        self.is_train = is_train

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

def get_data(dataset_names, model, tokenizer, batch_size):
    
    train_texts, train_questions, train_answers = [], [], []
    test_texts,  test_questions,  test_answers  = [], [], []
    
    print("Loading Datasets: ")
    
    for dataset_name in dataset_names:
        
        print(f"\tLoading {dataset_name}:")
        
        module = __map_name_to_module(dataset_name)
        train_split, test_split = module.get_data()
        
        print(f"\Succesfully loaded {dataset_name}:")
        
        module_train_texts, module_train_questions, module_train_answers = train_split 
        module_test_texts,  module_test_questions,  module_test_answers  = test_split 
        
        train_texts.extend(module_train_texts)
        train_questions.extend(module_train_questions)
        train_answers.extend(module_train_answers)
        
        test_texts.extend(module_test_texts)
        test_questions.extend(module_test_questions)
        test_answers.extend(module_test_answers)
        
        print(f"\t\tTrain texts {len(module_train_texts)}:")
        print(f"\t\tTrain questions {len(module_train_questions)}:")
        print(f"\t\tTrain answers {len(module_train_answers)}:")
        
        print(f"\t\tTest texts {len(module_test_texts)}:")
        print(f"\t\tTest questions {len(module_test_questions)}:")
        print(f"\t\tTest answers {len(module_test_answers)}:")
        
        __correct_span_errors(module_train_answers, module_train_texts)
        __correct_span_errors(module_test_answers, module_test_texts)
        
    print("\n\t\tTotal combined statistics")
        
    print(f"\t\tTrain texts {len(train_texts)}:")
    print(f"\t\tTrain questions {len(train_questions)}:")
    print(f"\t\tTrain answers {len(train_answers)}:")
    
    print(f"\t\tTest texts {len(test_texts)}:")
    print(f"\t\tTest questions {len(test_questions)}:")
    print(f"\t\tTest answers {len(test_answers)}:")
        
    train_encodings  = tokenizer(train_texts, train_questions, truncation=True, padding=True)
    test_encodings   = tokenizer(test_texts, test_questions, truncation=True, padding=True)
    
    __add_token_positions(train_encodings, tokenizer, module_train_answers)
    __add_token_positions(test_encodings, tokenizer, module_test_answers) 
        
    train_dataset = SquadDataset(train_encodings, is_train = True)
    test_dataset  = SquadDataset(test_encodings, is_train = False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
    
    test_data_raw = (test_texts, test_questions, test_answers)
    return train_loader, test_loader, test_data_raw