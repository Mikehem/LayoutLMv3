import random
import torch
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_optimizer(model, config):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": float(config['weight_decay']),
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    learning_rate = float(config['learning_rate'])
    adam_epsilon = float(config['adam_epsilon'])
    
    return AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

def get_scheduler(optimizer, config, total_steps):
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config['warmup_steps']),
        num_training_steps=total_steps
    )
