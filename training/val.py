import torch 
from tqdm import tqdm
from ..validation import evaluate_model

device = torch.device('cuda:0' if torch.cuda.is_available()
                      else 'cpu')


def validate(model, tokenizer, val_loader,  val_texts, val_questions, val_answers):
    model.eval()
    
    pbar = tqdm(total = len(val_loader))
    
    total_loss = 0
    
    for batch_idx, batch in enumerate(val_loader):
    
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = outputs[0]
            # Find the total loss
            total_loss += loss.item()
            
        pbar.set_postfix({'Batch': batch_idx+1, 'Loss': round(loss.item(),3)}, refresh=True)

    total_loss /= len(val_loader)
    
    metrics_dict = evaluate_model(model, tokenizer, val_texts, val_questions, val_answers)
    
    model.train()
    
    return total_loss, metrics_dict