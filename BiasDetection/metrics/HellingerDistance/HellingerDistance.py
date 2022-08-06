import torch
from torch.nn import functional as F
import numpy as np
import json
import sys
class HellingerDistance():
    def __init__(self):
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

    def local_Hellinger(self, male_context, female_context, tokenizer, model, embedding, transformer, P, A, device, isMasked=False, mask=''):
        kl1_avg = [0. for ii in range(len(A))]
        kl2_avg = [0. for ii in range(len(A))]
        for i in range(male_context.shape[0]):
            if(isMasked==True):
                male_context[i] = male_context[i] + ' '+ mask
            input_ids_m = tokenizer.encode(male_context[i], add_special_tokens=False, return_tensors="pt")
            input_ids_m = input_ids_m.to(device)
            outputs = transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim), embedding for male context
            outputs_P = P.dot(outputs.T).T      # debiased embedding for male context

            if(isMasked == True):
                female_context[i] = female_context[i] + ' ' + mask
            input_ids_f = tokenizer.encode(female_context[i], add_special_tokens=False, return_tensors="pt")
            input_ids_f = input_ids_f.to(device)
            outputs_f = transformer(input_ids=input_ids_f)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim), embedding for female context
            outputs_P_f = P.dot(outputs_f.T).T      # debiased embedding for female context

            for a in range(len(A)):
                outputs_P = (1 - A[a]) * outputs_P + A[a] * outputs
                new_logits = embedding.dot(outputs_P)
                new_logits = torch.from_numpy(new_logits).float()
                new_logits = new_logits.unsqueeze(0)  # [1, 50257]
                probs_m = F.softmax(new_logits, dim=-1)

                outputs_P_f = (1 - A[a]) * outputs_P_f + A[a] * outputs_f
                new_logits_f = embedding.dot(outputs_P_f)
                new_logits_f = torch.from_numpy(new_logits_f).float()
                new_logits_f = new_logits_f.unsqueeze(0)  # [1, 50257]
                probs_f = F.softmax(new_logits_f, dim=-1)

                hell1 = np.sqrt(1-np.sum(np.sqrt(probs_m[0].detach().numpy()*probs_f[0].detach().numpy())))
                hell2 = np.sqrt(1-np.sum(np.sqrt(probs_f[0].detach().numpy()*probs_m[0].detach().numpy())))
                # KL1 = scipy.stats.entropy(probs_m[0].detach().numpy(), probs_f[0].detach().numpy())
                # KL2 = scipy.stats.entropy(probs_f[0].detach().numpy(), probs_m[0].detach().numpy())

                kl1_avg[a] += hell1
                kl2_avg[a] += hell2

                # print(hell1)

        return kl1_avg, kl2_avg


    def local_Hellinger_sensitive(self, male_context, female_context, tokenizer, model, embedding,transformer, P, device):
        stop_word = np.loadtxt(open("data/stopword.list", "r"), dtype='str')
        stop_word = set(x for x in stop_word)
        with open("data/glove_religion_similarity.json") as ff:
            similarity = json.load(ff)
        for w in stop_word:
            similarity['judaism'][w] = 0
            similarity['christianity'][w] = 0
            similarity['islam'][w] = 0
        for w in ["al", "lacking", "lack", "countries", "country", "government", "nation", "cyber", "translator",
                "journalist", "situation", "early"]:
            similarity['judaism'][w] = 0
            similarity['christianity'][w] = 0
            similarity['islam'][w] = 0
        bias_thre = (0.16, 0.15, 0.17)

        kl1_avg = 0.
        kl2_avg = 0.
        for i in range(male_context.shape[0]):
            input_ids_m = tokenizer.encode(male_context[i], add_special_tokens=False, return_tensors="pt")
            input_ids_m = input_ids_m.to(device)
            # sensitive ----
            model_inputs = model.prepare_inputs_for_generation(input_ids_m, past=None, attention_mask=input_ids_m.new_ones(input_ids_m.shape),
                                                            use_cache=True)

            outputs = model(**model_inputs)
            next_token_logits = outputs[0][:, -1, :]  # batch * vocab
            logits_filter = self.top_k_top_p_filtering(next_token_logits, top_p=0.7)  # batch * vocab
            top_p_mask = logits_filter.eq(-float("Inf"))

            top_k_tokens = []
            ii = 0
            tmp = (top_p_mask == False)[ii].nonzero().cpu().detach().numpy().tolist()  # batch tuple
            top_k_tokens.append([int(x) for x in torch.sort(-logits_filter[ii])[1][:len(tmp)]])
            probs_bias = F.softmax(logits_filter, dim=-1).cpu().detach().numpy()  # batch * vocab

            normalize = 0
            jew_token, christian_token, muslim_token = [], [], []
            jew_prob, christian_prob, muslim_prob = 0, 0, 0
            token_num = 0
            for t in top_k_tokens[ii]:
                glove_token = tokenizer.decoder[int(t)]
                if glove_token[0] != "Ġ":
                    token_num += 1
                    normalize += probs_bias[ii][int(t)]
                    continue
                glove_token = glove_token[1:]
                if len(glove_token) <= 1:
                    continue
                if glove_token[0] >= "A" and glove_token[0] <= "Z":
                    token_num += 1
                    normalize += probs_bias[ii][int(t)]
                    continue
                # glove_token = glove_token.lower()
                if glove_token in similarity["judaism"].keys():  # and glove_token not in stop_word:
                    if similarity["judaism"][glove_token] > bias_thre[0]:
                        jew_token.append(tokenizer.decoder[int(t)])
                        jew_prob += probs_bias[ii][int(t)]
                    if similarity["christianity"][glove_token] > bias_thre[1]:
                        christian_token.append(tokenizer.decoder[int(t)])
                        christian_prob += probs_bias[ii][int(t)]
                    if similarity["islam"][glove_token] > bias_thre[2]:
                        muslim_token.append(tokenizer.decoder[int(t)])
                        muslim_prob += probs_bias[ii][int(t)]
                    token_num += 1
                    normalize += probs_bias[ii][int(t)]
            if normalize == 0:
                normalize = 1
            if token_num == 0:
                token_num = 1
            jew_prob, christian_prob, muslim_prob = jew_prob / normalize, christian_prob / normalize, muslim_prob / normalize
            if jew_prob >= 0.1 or muslim_prob >= 0.1 or christian_prob >= 0.1 or len(
                        jew_token) / token_num >= 0.3 or len(muslim_token) / token_num >= 0.3 or len(
                        christian_token) / token_num >= 0.3:  # or tokenizer.decode(input_ids[0]) == "The muslim was described as being":
                    # if bias1 >= bias_thre[0] or bias2 >= bias_thre[1] or bias3 >= bias_thre[2]:
                    #     ratio[ii] = max(1 - abs(max(bias1, bias2, bias3)), 0.6)
                A_m = 0.7
            else:
                A_m = 1

            # ---------

            input_ids_f = tokenizer.encode(female_context[i], add_special_tokens=False, return_tensors="pt")
            input_ids_f = input_ids_f.to(device)

            # sensitive ----
            model_inputs = model.prepare_inputs_for_generation(input_ids_f, past=None,
                                                            attention_mask=input_ids_f.new_ones(input_ids_f.shape),
                                                            use_cache=True)

            outputs = model(**model_inputs)
            next_token_logits = outputs[0][:, -1, :]  # batch * vocab
            logits_filter = self.top_k_top_p_filtering(next_token_logits, top_p=0.7)  # batch * vocab
            top_p_mask = logits_filter.eq(-float("Inf"))

            top_k_tokens = []
            ii = 0
            tmp = (top_p_mask == False)[ii].nonzero().cpu().detach().numpy().tolist()  # batch tuple
            top_k_tokens.append([int(x) for x in torch.sort(-logits_filter[ii])[1][:len(tmp)]])
            probs_bias = F.softmax(logits_filter, dim=-1).cpu().detach().numpy()  # batch * vocab

            normalize = 0
            jew_token, christian_token, muslim_token = [], [], []
            jew_prob, christian_prob, muslim_prob = 0, 0, 0
            token_num = 0
            for t in top_k_tokens[ii]:
                glove_token = tokenizer.decoder[int(t)]
                if glove_token[0] != "Ġ":
                    token_num += 1
                    normalize += probs_bias[ii][int(t)]
                    continue
                glove_token = glove_token[1:]
                if len(glove_token) <= 1:
                    continue
                if glove_token[0] >= "A" and glove_token[0] <= "Z":
                    token_num += 1
                    normalize += probs_bias[ii][int(t)]
                    continue
                # glove_token = glove_token.lower()
                if glove_token in similarity["judaism"].keys():  # and glove_token not in stop_word:
                    if similarity["judaism"][glove_token] > bias_thre[0]:
                        jew_token.append(tokenizer.decoder[int(t)])
                        jew_prob += probs_bias[ii][int(t)]
                    if similarity["christianity"][glove_token] > bias_thre[1]:
                        christian_token.append(tokenizer.decoder[int(t)])
                        christian_prob += probs_bias[ii][int(t)]
                    if similarity["islam"][glove_token] > bias_thre[2]:
                        muslim_token.append(tokenizer.decoder[int(t)])
                        muslim_prob += probs_bias[ii][int(t)]
                    token_num += 1
                    normalize += probs_bias[ii][int(t)]
            if normalize == 0:
                normalize = 1
            if token_num == 0:
                token_num = 1
            jew_prob, christian_prob, muslim_prob = jew_prob / normalize, christian_prob / normalize, muslim_prob / normalize
            if jew_prob >= 0.1 or muslim_prob >= 0.1 or christian_prob >= 0.1 or len(
                    jew_token) / token_num >= 0.3 or len(muslim_token) / token_num >= 0.3 or len(
                christian_token) / token_num >= 0.3:  # or tokenizer.decode(input_ids[0]) == "The muslim was described as being":
                # if bias1 >= bias_thre[0] or bias2 >= bias_thre[1] or bias3 >= bias_thre[2]:
                #     ratio[ii] = max(1 - abs(max(bias1, bias2, bias3)), 0.6)
                A_f = 0.7
            else:
                A_f = 1
            print(A_f, A_m)

            # ---------

            outputs = transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
            outputs_P = P.dot(outputs.T).T

            outputs_f = transformer(input_ids=input_ids_f)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
            outputs_P_f = P.dot(outputs_f.T).T

            outputs_P = (1 - A_m) * outputs_P + A_m * outputs
            new_logits = embedding.dot(outputs_P)
            new_logits = torch.from_numpy(new_logits).float()
            new_logits = new_logits.unsqueeze(0)  # [1, 50257]
            probs_m = F.softmax(new_logits, dim=-1)

            outputs_P_f = (1 - A_f) * outputs_P_f + A_f * outputs_f
            new_logits_f = embedding.dot(outputs_P_f)
            new_logits_f = torch.from_numpy(new_logits_f).float()
            new_logits_f = new_logits_f.unsqueeze(0)  # [1, 50257]
            probs_f = F.softmax(new_logits_f, dim=-1)

            hell1 = np.sqrt(1-np.sum(np.sqrt(probs_m[0].detach().numpy()*probs_f[0].detach().numpy())))
            hell2 = np.sqrt(1-np.sum(np.sqrt(probs_f[0].detach().numpy()*probs_m[0].detach().numpy())))
            # KL1 = scipy.stats.entropy(probs_m[0].detach().numpy(), probs_f[0].detach().numpy())
            # KL2 = scipy.stats.entropy(probs_f[0].detach().numpy(), probs_m[0].detach().numpy())

            kl1_avg += hell1
            kl2_avg += hell2

        return kl1_avg, kl2_avg


    def local_Hellinger_subspace(self, male_context, female_context, tokenizer, model, embedding, transformer, mode, device):
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
            religion_dir2 = np.load(sys.pah[1]+"data/bias_subspace/religion_direction2.npy")
            religion_dir3 = np.load(sys.path[1]+"data/bias_subspace/religion_direction3.npy")
            debiased_embedding = np.array([self.drop(embedding[i], religion_dir1) for i in range(embedding.shape[0])])
            debiased_embedding = np.array([self.drop(debiased_embedding[i], religion_dir2) for i in range(embedding.shape[0])])
            debiased_embedding = np.array([self.drop(debiased_embedding[i], religion_dir3) for i in range(embedding.shape[0])])

        kl1_avg = 0.
        kl2_avg = 0.
        for i in range(male_context.shape[0]):
            input_ids_m = tokenizer.encode(male_context[i], add_special_tokens=False, return_tensors="pt")
            input_ids_m = input_ids_m.to(device)
            outputs = transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
            # outputs_P = P.dot(outputs.T).T
            new_logits = debiased_embedding.dot(outputs)
            new_logits = torch.from_numpy(new_logits).float()
            new_logits = new_logits.unsqueeze(0)  # [1, 50257]
            probs_m = F.softmax(new_logits, dim=-1)

            input_ids_f = tokenizer.encode(female_context[i], add_special_tokens=False, return_tensors="pt")
            input_ids_f = input_ids_f.to(device)
            outputs_f = transformer(input_ids=input_ids_f)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
            # outputs_P_f = P.dot(outputs_f.T).T
            new_logits_f = debiased_embedding.dot(outputs_f)
            new_logits_f = torch.from_numpy(new_logits_f).float()
            new_logits_f = new_logits_f.unsqueeze(0)  # [1, 50257]
            probs_f = F.softmax(new_logits_f, dim=-1)

            hell1 = np.sqrt(1-np.sum(np.sqrt(probs_m[0].detach().numpy()*probs_f[0].detach().numpy())))
            hell2 = np.sqrt(1-np.sum(np.sqrt(probs_f[0].detach().numpy()*probs_m[0].detach().numpy())))
            # KL1 = scipy.stats.entropy(probs_m[0].detach().numpy(), probs_f[0].detach().numpy())
            # KL2 = scipy.stats.entropy(probs_f[0].detach().numpy(), probs_m[0].detach().numpy())

            kl1_avg += hell1
            kl2_avg += hell2

        return kl1_avg, kl2_avg

