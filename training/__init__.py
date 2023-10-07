from .train import train_epoch
from .val import validate
from torch.optim import AdamW
import wandb
from transformers import get_linear_schedule_with_warmup

def run_training(train_loader, test_loader, test_data_raw, model, tokenizer, epochs, lr, run_name):
    
    num_training_steps = epochs * len(train_loader)  # total number of training steps 
    num_warmup_steps = num_training_steps // 10  # warmup for the first 10% of steps
    
    optim = AdamW(model.parameters(), lr=lr)
    
    scheduler = get_linear_schedule_with_warmup(optim, 
                                            num_warmup_steps=num_warmup_steps, 
                                            num_training_steps=num_training_steps)
    
    test_texts, test_questions, test_answers = test_data_raw

    highest_f1 = 0
    
    for epoch in range(epochs):
        print(f"\n****** epoch {epoch + 1}/{epochs} ********\n")
        train_loss = train_epoch(model, train_loader, optim, scheduler)
        val_loss, metrics_dict = validate(model, tokenizer, test_loader,  test_texts, test_questions, test_answers)
        
        loss_dict = {'train_loss': train_loss, 'val_loss': val_loss}

        f1_score = metrics_dict['f1']
        if f1_score > highest_f1:
            highest_f1 = f1_score
            model.save_pretrained(run_name + "_f1_" + str(round(f1_score, 4)))


        wandb.log({**loss_dict, **metrics_dict})
        
    
        
    
        
        
        