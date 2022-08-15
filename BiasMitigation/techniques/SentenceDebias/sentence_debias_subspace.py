import argparse
import os

import torch
import transformers
import sys
sys.path.insert(1, 'C:/Users/hrish/Documents/Purdue/Summer 22/Language Bias/Bias Detection/bias-bench-main')
from techniques.SentenceDebias.sentence_debias import load_sentence_debias_data
from techniques.SentenceDebias.sentence_subspaces import (
    compute_gender_subspace,
    compute_race_subspace,
    compute_religion_subspace,
)
import models

def sentence_debias(model, tokenizer, model_class, dataset, dataset_name, bias_type, num_classifiers=1, output_dir='BiasMitigation/SentenceDebias', seed=0, batch_size=32):
    print("Computing bias subspace:")
    print(f" - persistent_dir: {output_dir}")
    print(f" - model_name_or_path: {model_class}")
    print(f" - bias_type: {bias_type}")
    print(f" - batch_size: {batch_size}")

    # Get the data to compute the SentenceDebias bias subspace.
    data = load_sentence_debias_data(dataset=dataset_name,
        persistent_dir=dataset, bias_type=bias_type,
    )

    # Load model and tokenizer.
    #model = getattr(models, model)(args.model_name_or_path)
    model.eval()
    #tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Specify a padding token for batched SentenceDebias subspace computation for
    # GPT2.
    if model_class == "gpt2":
        tokenizer.pad_token = tokenizer.eos_token

    if bias_type == "gender":
        bias_direction = compute_gender_subspace(
            data, model, tokenizer, batch_size=batch_size
        )
    elif bias_type == "race":
        bias_direction = compute_race_subspace(
            data, model, tokenizer, batch_size=batch_size
        )
    else:
        bias_direction = compute_religion_subspace(
            data, model, tokenizer, batch_size=batch_size
        )

    print(
        f"Saving computed PCA components to: {output_dir}/results/sent_debias_{model_class}_{bias_type}.pt."
    )
    os.makedirs(f"{output_dir}/results/", exist_ok=True)
    torch.save(
        bias_direction, f"{output_dir}/results/sent_debias_{model_class}_{bias_type}.pt"
    )
    return bias_direction
