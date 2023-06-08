from .train import train_epoch
from .val import validate
from torch.optim import AdamW
import wandb

def run_training(train_loader, test_loader, test_data_raw, model, tokenizer, epochs, lr):
    
    optim = AdamW(model.parameters(), lr=lr)
    
    test_texts, test_questions, test_answers = test_data_raw
    
    for epoch in range(epochs):
        print(f"\n****** epoch {epoch + 1}/{epochs} ********\n")
        train_loss = train_epoch(model, train_loader, optim)
        val_loss, metrics_dict = validate(model, tokenizer, test_loader,  test_texts, test_questions, test_answers)
        
        loss_dict = {'train_loss': train_loss, 'val_loss': val_loss}
        wandb.log({**loss_dict, **metrics_dict})
        
    
        
    
        
        
        