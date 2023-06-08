import torch
import wandb 
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available()
                      else 'cpu')

def train_epoch(model, train_loader, optim):

    # Set model in train mode
    model.train()
        
    total_loss = 0

    pbar = tqdm(train_loader)
    for batch in pbar: 
        
        optim.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        wandb.log({'train_loss': loss.item()})
        # do a backwards pass 
        loss.backward()
        
        # update the weights
        optim.step()
        # Find the total loss
        total_loss += loss.item()
        
        pbar.set_postfix({'batch loss': loss.item()})
        
        wandb.log({'batch_loss': loss.item()})

    total_loss /= len(train_loader)
    
    return total_loss