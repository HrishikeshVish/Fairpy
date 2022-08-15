# import IPython; IPython.embed(); exit(1)

#import ruamel.yaml as yaml
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler
from transformers import AutoTokenizer
from datasets import load_dataset

from techniques.DiffPruning.src.model import DiffNetwork
from techniques.DiffPruning.src.training_logger import TrainLogger
from techniques.DiffPruning.src.metrics import accuracy


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET = "glue_sentiment"

def get_ds(tokenizer) -> TensorDataset:
    
    ds = load_dataset("glue", "sst2", cache_dir="cache")
    return ds.map(
        lambda x: tokenizer(x["sentence"], padding="max_length", max_length=128, truncation=True),
        batched=True,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset"
    )
    

def get_ds_part(ds, part) -> TensorDataset:
    _ds = ds[part]
    return TensorDataset(
        torch.tensor(_ds["input_ids"], dtype=torch.long),
        torch.tensor(_ds["token_type_ids"], dtype=torch.long),
        torch.tensor(_ds["attention_mask"], dtype=torch.long),
        torch.tensor(_ds["label"], dtype=torch.float)
    )

def batch_fn(batch):
    input_ids, token_type_ids, attention_masks, labels = [torch.stack(l) for l in zip(*batch)]
    x = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_masks
    }
    return x, labels


def DiffPruning(model_name='bert-base-uncased', batch_size=128, structured_diff_pruning=True, alpha_init=5, concrete_lower=-1.5, concrete_upper=1.5, gradient_accumulation_steps=1, 
                diff_pruning=True, num_epochs_finetune=1, num_epochs_fixmask=1, weight_decay=0.0, learning_rate=5e-5, learning_rate_alpha=0.1, adam_epsilon=1e-8,
                warmup_steps=0, sparsity_pen=1.25e-7, max_grad_norm=1.0, fixmask_pct=0.01, logging_step=5, output_dir='checkpoints', log_dir='logs'):    
        
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    ds = get_ds(tokenizer)
    
    pred_fn = lambda x: (x > 0.).long()
    loss_fn = lambda x, y: torch.nn.BCEWithLogitsLoss()(x.flatten(), y)
    metrics = {
        "acc": lambda x, y: accuracy(pred_fn(x), y)
    }
   
    ds_train = get_ds_part(ds, "train")
    train_loader = DataLoader(ds_train, sampler=RandomSampler(ds_train), batch_size=batch_size, collate_fn=batch_fn)
    ds_eval = get_ds_part(ds, "validation")
    eval_loader = DataLoader(ds_eval, sampler=SequentialSampler(ds_eval), batch_size=batch_size, collate_fn=batch_fn)

    logger_name = "_".join([
        "diff_pruning" if diff_pruning else "finetuning",
        "fixmask" if num_epochs_fixmask > 0 else "no_fixmask",
        model_name.split('/')[-1],
        DATASET,
        str(batch_size),
        str(learning_rate)
    ])
    train_logger = TrainLogger(
        log_dir = log_dir,
        logger_name = logger_name,
        logging_step = logging_step
    )

    trainer = DiffNetwork(1, model_name)
    trainer.to(DEVICE)

    encoder, classifier = trainer.fit(
        train_loader,
        eval_loader,
        train_logger,
        loss_fn,
        metrics,
        num_epochs_finetune,
        num_epochs_fixmask,
        diff_pruning,
        alpha_init,
        concrete_lower,
        concrete_upper,
        structured_diff_pruning,
        sparsity_pen,
        fixmask_pct,
        weight_decay,
        learning_rate,
        learning_rate_alpha,
        adam_epsilon,
        warmup_steps,
        gradient_accumulation_steps,
        max_grad_norm,
        output_dir
    )
    return encoder, classifier


