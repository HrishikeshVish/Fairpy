import argparse
import os

import torch
import transformers

from techniques.NullSpaceProjection.inlp import load_inlp_data
from techniques.NullSpaceProjection.context_nullspace_projection import compute_projection_matrix

def ComputeProjectionMatrix(model, tokenizer, model_class, dataset, dataset_name, bias_type, num_classifiers=1, output_dir='BiasMitigation/NullSpaceProjection', seed=0):

    print("Computing projection matrix:")
    print(f" - persistent_dir: {output_dir}")
    print(f" - model_name_or_path: {model_class}")
    print(f" - bias_type: {bias_type}")
    print(f" - n_classifiers: {num_classifiers}")
    print(f" - seed: {seed}")

    # Load data for INLP classifiers.
    data = load_inlp_data('BiasMitigation', bias_type, dataset, seed=seed)

    # Load model and tokenizer.
    #model = getattr(models, args.model)(args.model_name_or_path)
    model.eval()
    #tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    projection_matrix = compute_projection_matrix(
        model,
        tokenizer,
        data,
        bias_type=bias_type,
        n_classifiers=num_classifiers,
    )

    print(
        f"Saving computed projection matrix to: {output_dir}/results/nsp_{bias_type}_{dataset_name}.pt"
    )
    os.makedirs(f"{output_dir}/results/", exist_ok=True)
    torch.save(
        projection_matrix,
        f"{output_dir}/results/nsp_{bias_type}_{dataset_name}.pt",
    )
    return projection_matrix
