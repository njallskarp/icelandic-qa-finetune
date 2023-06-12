import os 
import requests
import json

def get_data():
    
    test_texts,   train_texts = [], [] 
    test_questions, train_questions = [], [] 
    test_answers, train_answers = [], []
    
    PROJECT_IDS = [1, 4, 5]
    
    
    seen_qs = set()
    seen_as = set()
    
    for project_id in PROJECT_IDS:
        url = f"https://labeling.gameqa.app/api/projects/{project_id}/export?exportType=JSON"
        # Define the URL and headers
        headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://labeling.gameqa.app/projects/5/data/export",
            "Sec-Ch-Ua": '"Google Chrome";v="113", "Chromium";v="113", "Not-A.Brand";v="24"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": "macOS",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "same-origin",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
        }

        # Define the cookies
        cookies = {
            "i18n_redirected": "en", 
            "_ga_2X09492CEB": "GS1.1.1685019468.27.1.1685019468.0.0.0", 
            "_ga": "GA1.2.1330172852.1682087891", 
            "csrftoken": "vFXEabLncincpaRN9yCS6o8l3qovl4cTvL6ITC2JFMLFCl8eeWklKG6GHPNlqdZA", 
            "sessionid": ".eJxVj0tuxCAQRO_CeowaGgx4mX3OYDUf22QsGBlbyke5e-xoNrOsrqpX6h925MgGppSUIVjdEQnbKdn7zgqSHU6QJHjvXNTsxlZq-7jWORc2iN7qXrgeDUfAs6JvbKRjX8ajpW38xwr2cvMU7qlcRvygMlceatm37PkV4U-38fca0_r2zL4AFmrL2UabcBI-gUa0RnoQMClnlJ-oFxA0CtIqAJK2FBUmD9pEkKTRKEfCndC6zVTyN-25lvFxP985h1pq7dLp85G3LzYolADw-wfOXFo4:1q7CV6:yrVw6xYWNwLTwoXfbO8rCTkWJM9eOQ0o7bdUX0f7gzY"
        }

        # Make the GET request
        response = requests.get(url, headers=headers, cookies=cookies)

        
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
            start_idx = annotation['start']
            end_idx = annotation['end']

            q = record['meta']['question']
            if q in seen_qs:
                continue
            seen_qs.add(q)
            p = record['meta']['paragraph']
            split = record['meta']['split']
            a = p[start_idx:end_idx]

            if len(p.split(" ")) > 300:
                continue
            
            answer_key = (p, start_idx, end_idx)
            if answer_key in seen_as:
                continue
            seen_as.add(answer_key)
    
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
    
