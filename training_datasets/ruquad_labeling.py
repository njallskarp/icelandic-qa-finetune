import os 
import requests
import json

from config import LABELSTUDIO_TOKEN


def get_data():
    
    test_texts,   train_texts = [], [] 
    test_questions, train_questions = [], [] 
    test_answers, train_answers = [], []
    
    PROJECT_IDS = [1, 4, 5]
    
    
    seen_qs = set()
    seen_as  = set()
    
    for project_id in PROJECT_IDS:
        url = f"https://labeling.gameqa.app/api/projects/{project_id}/export?exportType=JSON"
        # Define the URL and headers
        headers = {"Authorization": f"Token {LABELSTUDIO_TOKEN}"}
        # Make the GET request
        response = requests.get(url, headers=headers)
        response.encoding = "utf-8"

        
        # Print the response
        records = json.loads(response.text)

        for record in records:
            if len(record['annotations']) == 0:
                continue
            annotation = record["annotations"][0]['result']
            if not annotation:
                continue
            annotation = annotation[0]["value"]
            if 'labels' not in annotation:
                continue
            label = annotation['labels'][0]
            if label == "Archive":
                continue
            if record['meta']['type'] == "ANSWERED_YES_NO":
                continue

            p = record['meta']['paragraph']
            
            start_idx = record['meta']['start']
            end_idx = record['meta']['end']

            answer_key = (p, start_idx, end_idx)
            q = record['meta']['question']
            if q in seen_qs or answer_key in seen_as:
                seen_qs.add(q)
                seen_as.add(answer_key)
                continue
            seen_qs.add(q)
            seen_as.add(answer_key)
            split = record['meta']['split']
            a = p[start_idx:end_idx]

            if len(p.split(" ")) > 300:
                continue
            
    
            if split == "train":
                train_texts.append(p)
                train_questions.append(q)
                train_answers.append({
                    'answer_end':   end_idx,
                    'answer_start': start_idx,
                    'text':         a,
                })
            else:
                test_texts.append(p)
                test_questions.append(q)
                test_answers.append({
                    'answer_end':   end_idx,
                    'answer_start': start_idx,
                    'text':         a,
                })

    
    DEST = "./datafiles/ruquad_1_unstandardized.zip"
    URL  = "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/311"
    
    os.system(f"curl --output {DEST} {URL}")
    
    return (train_texts, train_questions, train_answers),  (test_texts, test_questions, test_answers)
    
