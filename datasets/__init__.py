from . import ruquad_unstandardized_1
from .config import ALLOWED_DATASETS
from transformers import AutoTokenizer,AdamW,BertForQuestionAnswering
from torch.util.data import DataLoader, Dataset

def __map_name_to_module(name):
    
    if name not in ALLOWED_DATASETS:
        raise ModuleNotFoundError(f"The dataset requested '{name}' not found")
    if name == "ruquad_unstandardized_1":
        return ruquad_unstandardized_1
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
            
def __add_token_positions(encodings, answers):
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
            end = encodings.char_to_token(i, answers[i]['answer_end'] - 1)
        if end  is None:
            end = tokenizer.model_max_length
            
        start_pos.append(start)
        end_pos.append(end)
    encodings.update({'start_positions': start, 'end_positions': end})
    
class SquadDataset(torch.utils.data.Dataset):
    """
    Code reused from open source colab notebook:
    https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset/blob/main/Fine_Tuning_Bert.ipynb
    """
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
      return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

def get_data(dataset_names, model, tokenizer, batch_size):
    
    train_texts, train_questions, train_answers = [], [], []
    test_texts, test_questions, test_answers = [], [], []
    
    for dataset_name in dataset_names:
        module = __map_name_to_module(dataset_name)
        train_split, test_split = module.get_data()
        
        module_train_texts, module_train_questions, module_train_answers = train_split 
        module_test_texts, module_test_questions, module_test_answers = test_split 
        
        train_texts.extend(module_train_texts)
        train_questions.extend(module_train_questions)
        train_answers.extend(module_train_answers)
        
        test_texts.extend(module_test_texts)
        test_questions.extend(module_test_questions)
        test_answers.extend(module_test_answers)
        
        __correct_span_errors(module_train_answers, module_train_texts)
        __correct_span_errors(module_test_answers, module_test_texts)
        
        module_train_encodings = tokenizer(module_train_texts, module_train_questins, truncation=True, padding=True)
        module_test_encodings   = tokenizer(module_test_texts, module_test_questions, truncation=True, padding=True)
        
        add_token_positions(module_train_encodings, module_train_answers)
        add_token_positions(module_test_encodings, module_test_answers)
        
        train_texts.extend(module_train_texts)
        train_questions.extend(module_train_questions)
        train_answers.extend(module_train_answers)
        
        test_texts.extend(module_test_texts)
        test_questions.extend(module_test_questions)
        test_answers.extend(module_test_answers)
        
        
    train_dataset = SquadDataset(train_encodings)
    test_dataset  = SquadDataset(test_encodings)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
    
    test_data_raw = (test_texts, test_questions, test_answers)
    return train_loader, test_loader, test_data_raw