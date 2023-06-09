import os 
import requests
import json
import datasets


def get_data():
    
    test_texts,   train_texts = [], [] 
    test_questions, train_questions = [], [] 
    test_answers, train_answers = [], []
    
    dataset = load_dataset('vesteinn/icelandic-qa-NQiI')
    
    for row in dataset['train']:
        p = row['context']
        q = row['question']
        if not row['answers']['answer_start']:
            continue
        a = {
            'answer_start': row['answers']['answer_start'][0],
            'text': row['answers']['text'][0]
        }
        train_texts.append(p)
        train_questions.append(q)
        train_answers.append(a)
    
    for row in dataset['validation']:
        p = row['context']
        q = row['question']
        if not row['answers']['answer_start']:
            continue
        a = {
            'answer_start': row['answers']['answer_start'][0],
            'text': row['answers']['text'][0]
        }
        test_texts.append(p)
        test_questions.append(q)
        test_answers.append(a)


    
    DEST = "./datafiles/ruquad_1_unstandardized.zip"
    URL  = "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/311"
    
    os.system(f"curl --output {DEST} {URL}")
    
    return (train_texts, train_questions, train_answers),  (test_texts, test_questions, test_answers)
    
