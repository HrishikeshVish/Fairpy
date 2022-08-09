import torch
from torch.nn import functional as F
import sys
import numpy as np
class WeatProbability():
    def __init__(self, model, tokenizer, device, model_class, model_type, mask_token='[MASK]', dataset=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_class = model_class
        self.mask_token = mask_token
        self.dataset = dataset
        self.model_type = model_type
        return
    def dropspace(self,u, V):
        # u, V = u.detach().numpy(), V.detach().numpy()
        norm_sqrd = np.sum(V*V, axis=-1)
        vecs = np.divide(V@u, norm_sqrd)[:, None] * V
        subspace = np.sum(vecs, axis=0)
        return u - subspace

    def drop_bias(self,u, v):
        # return u - torch.ger(torch.matmul(u, v), v) / v.dot(v)
        projection = u.dot(v) * v / np.linalg.norm(v)
        return u - projection

    def drop(self,u, v):
        return u - v * u.dot(v) / v.dot(v)
    def top_k_top_p_filtering(self,
        logits,    # (1, 50257)
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
        ):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (batch size, vocabulary size)
                if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
                Make sure we keep at least min_tokens_to_keep per batch example in the output
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits

    def weat_true_label(self, weat_dataset, weat_pos, model, embedding, transformer, A, P, p, device, topk=False):
        if topk:
            weat_topk = 0.
            count = 0
            for i in range(len(weat_dataset)):
                input_ids_m = weat_dataset[i][1]
                input_ids_m = input_ids_m.to(device)
                outputs = transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
                logits = embedding.dot(outputs)
                logits_filter = torch.from_numpy(logits).float().clone()
                logits_filter = logits_filter.unsqueeze(0)
                logits_filter = self.top_k_top_p_filtering(logits_filter, top_p=p)
                top_p_mask = logits_filter.eq(-float("Inf"))

                outputs_P = P.dot(outputs.T).T
                new_logits = embedding.dot(outputs_P)
                logits_filter = torch.from_numpy(new_logits).float().clone()
                logits_filter = logits_filter.unsqueeze(0)
                logits_filter.masked_fill_(top_p_mask, -float("Inf"))
                probs_m = F.softmax(logits_filter, dim=-1).detach().numpy()

                weat_topk += probs_m[0][weat_pos[i]]
                count += 1
            return weat_topk / count

        weat_avg = [0. for ii in range(len(A))]
        count = 0
        for i in range(len(weat_dataset)):
            input_ids_m = weat_dataset[i][1]
            input_ids_m = input_ids_m.to(device)
            outputs = transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
            outputs_P = P.dot(outputs.T).T
            for a in range(len(A)):
                outputs_P = (1 - A[a]) * outputs_P + A[a] * outputs
                new_logits = embedding.dot(outputs_P)
                new_logits = torch.from_numpy(new_logits).float()
                new_logits = new_logits.unsqueeze(0)  # [1, 50257]
                probs_m = F.softmax(new_logits, dim=-1).detach().numpy()

                weat_avg[a] += probs_m[0][weat_pos[i]]
            count += 1
        return [x / count for x in weat_avg]

    def weat_true_label_sensitive(self, weat_dataset, weat_pos, model, embedding, transformer, mode, p, device, A, P, topk=False):
        if topk:
            weat_topk = 0.
            count = 0
            for i in range(len(weat_dataset)):
                input_ids_m = weat_dataset[i][1]
                input_ids_m = input_ids_m.to(device)
                outputs = transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
                logits = embedding.dot(outputs)
                logits_filter = torch.from_numpy(logits).float().clone()
                logits_filter = logits_filter.unsqueeze(0)
                logits_filter = self.top_k_top_p_filtering(logits_filter, top_p=p)
                top_p_mask = logits_filter.eq(-float("Inf"))

                outputs_P = P.dot(outputs.T).T
                new_logits = embedding.dot(outputs_P)
                logits_filter = torch.from_numpy(new_logits).float().clone()
                logits_filter = logits_filter.unsqueeze(0)
                logits_filter.masked_fill_(top_p_mask, -float("Inf"))
                probs_m = F.softmax(logits_filter, dim=-1).detach().numpy()

                weat_topk += probs_m[0][weat_pos[i]]
                count += 1
            return weat_topk / count

        weat_avg = [0. for ii in range(len(A))]
        count = 0
        for i in range(len(weat_dataset)):
            input_ids_m = weat_dataset[i][1]
            input_ids_m = input_ids_m.to(device)
            outputs = transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
            outputs_P = P.dot(outputs.T).T
            for a in range(len(A)):
                outputs_P = (1 - A[a]) * outputs_P + A[a] * outputs
                new_logits = embedding.dot(outputs_P)
                new_logits = torch.from_numpy(new_logits).float()
                new_logits = new_logits.unsqueeze(0)  # [1, 50257]
                probs_m = F.softmax(new_logits, dim=-1).detach().numpy()

                weat_avg[a] += probs_m[0][weat_pos[i]]
            count += 1
        return [x / count for x in weat_avg]

    def weat_true_label_subspace(self, weat_dataset, weat_pos, model, embedding, mode, transformer, p, device, A, P, topk=False):
        if mode[1] == "gender":
            if mode[0] == "direction":
                gender_direction = np.load(sys.path[1]+"data/bias_subspace/gpt2_gender_direction.npy")
                debiased_embedding = np.array([self.drop(embedding[i], gender_direction) for i in range(embedding.shape[0])])
            else:
                gender_direction = np.load(sys.path[1]+"data/bias_subspace/gpt2_gender_subspace.npy")
                debiased_embedding = np.array([self.dropspace(embedding[i], gender_direction) for i in range(embedding.shape[0])])
            # self.embedding.to(self.args.device)
        elif mode[1] == "religion":
            religion_dir1 = np.load(sys.path[1]+"data/bias_subspace/religion_direction1.npy")
            religion_dir2 = np.load(sys.path[1]+"data/bias_subspace/religion_direction2.npy")
            religion_dir3 = np.load(sys.path[1]+"data/bias_subspace/religion_direction3.npy")
            debiased_embedding = np.array([self.drop(embedding[i], religion_dir1) for i in range(embedding.shape[0])])
            debiased_embedding = np.array([self.drop(debiased_embedding[i], religion_dir2) for i in range(embedding.shape[0])])
            debiased_embedding = np.array([self.drop(debiased_embedding[i], religion_dir3) for i in range(embedding.shape[0])])

        weat_avg = 0.
        count = 0
        for i in range(len(weat_dataset)):
            input_ids_m = weat_dataset[i][1]
            input_ids_m = input_ids_m.to(device)
            outputs = transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
            # outputs_P = P.dot(outputs.T).T
            # for a in range(len(A)):
            #     outputs_P = (1 - A[a]) * outputs_P + A[a] * outputs
            new_logits = debiased_embedding.dot(outputs)
            new_logits = torch.from_numpy(new_logits).float()
            new_logits = new_logits.unsqueeze(0)  # [1, 50257]
            probs_m = F.softmax(new_logits, dim=-1).detach().numpy()

                # weat_avg[a] += probs_m[0][weat_pos[i]]
            weat_avg += probs_m[0][weat_pos[i]]
            count += 1
        return weat_avg / count
