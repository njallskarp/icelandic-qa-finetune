import string
import re
import collections
from transformers import pipeline
from tqdm import tqdm
import numpy as np

"""
The code in this cell is adapted from the official SQuAD validation script

https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
"""

def __normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
      regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
      return re.sub(regex, ' ', text)

    def white_space_fix(text):
      return ' '.join(text.split())

    def remove_punc(text):
      exclude = set(string.punctuation)
      return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
      return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def __get_tokens(s):
    if not s: return []
    return __normalize_answer(s).split()


def __span_comparison_helper(a_gold, a_pred):
    gold_toks = __get_tokens(a_gold)
    pred_toks = __get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    
    return num_same, pred_toks, gold_toks

def __recall(num_same, gold_toks):
    if len(gold_toks) == 0 or num_same == 0:
      return int(gold_toks == num_same)
    return 1.0 * num_same / len(gold_toks)

def __precision(num_same, pred_toks):
    if len(pred_toks) == 0:
      return 0
    return 1.0 * num_same / len(pred_toks) 

def __f1(num_same, pred_toks, gold_toks):
  
    if len(gold_toks) == 0 or len(pred_toks) == 0:
      return int(gold_toks == pred_toks)
    
    p = __precision(num_same, pred_toks)
    r = __recall(num_same, gold_toks)
    return 2*(p*r)/(p + r + 1e-8)

def recall(a_gold, a_pred):
    num_same, _, gold_toks = __span_comparison_helper(a_gold, a_pred)
    return __recall(num_same, gold_toks)

def precision(a_gold, a_pred):
    num_same, pred_toks, _ = __span_comparison_helper(a_gold, a_pred)
    return __precision(num_same, pred_toks)

def f1(a_gold, a_pred):
    vars = __span_comparison_helper(a_gold, a_pred)
    return __f1(*vars)

def evaluate_model(model, tokenizer, val_texts, val_queries, val_answers):
  i = 0

  nlp = pipeline('question-answering', model=model, tokenizer=tokenizer, device = 0)

  data = [(c, q, a) for c, q, a in zip(val_texts, val_queries, val_answers)]

  results = []

  for context, question, answer in data:
    res = nlp({
      'question': question,
      'context': context
    })
    
    answer_pred = res['answer']
    
    num_same, pred_toks, gold_toks = __span_comparison_helper(answer['text'], answer_pred)
    
    p  = __precision(num_same, pred_toks)
    r  = __recall(num_same, gold_toks)
    f1 = __f1(num_same, pred_toks, gold_toks)
    
    results.append({'precision': p, 'recall': r, 'f1': f1})
  
  avg_p = np.mean([_['precision'] for _ in results])
  avg_r = np.mean([_['recall'] for _ in results])
  avg_f = np.mean([_['f1'] for _ in results])
  
  return ({'precision': avg_p, 'recall': avg_r, 'f1': avg_f})
  