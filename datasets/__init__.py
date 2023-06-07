from . import ruquad_unstandardized_1
from .utils import ALLOWED_DATASETS
from transformers import AutoTokenizer,AdamW,BertForQuestionAnswering

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

def get_data(dataset_names, model, tokenizer):
    
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
        
        __correct_span_errors(train_answers, train_texts)
        __correct_span_errors(test_answers, test_texts)
        
    return (train_texts, train_questions, train_answers), (test_texts, test_questions, test_answers)