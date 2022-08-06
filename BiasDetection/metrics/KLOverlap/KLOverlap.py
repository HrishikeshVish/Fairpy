import numpy as np
import scipy
import torch
from torch.nn import functional as F
from sklearn.decomposition import PCA
import sys
class KLOverlap():
    def __init__(self):
        return
    def doPCA(self, pairs, num_components=10):
        matrix = []
        for a, b in pairs:
            center = (a + b) / 2
            norm_a = a - center
            norm_b = b - center
            norm_a, norm_b = norm_a.detach().numpy(), norm_b.detach().numpy()
            # norm_a, norm_b = norm_a/np.linalg.norm(norm_a), norm_b/np.linalg.norm(norm_b)
            matrix.append(norm_a)
            matrix.append(norm_b)
        matrix = np.array(matrix)
        pca = PCA(n_components=num_components, svd_solver="full")
        pca.fit(matrix)  # Produce different results each time...
        return pca
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

    def topk_kl_overlap(self, prompt_text, k, tokenizer, model, embedding, transformer, P, A, device):
        """
            :param prompt_text: a single prompt
            :param k: top k
            :param tokenizer: tokenizer
            :param model: gpt2 or other language model
            :param embedding: gpt2 word embedding
            :param P: nullspace matrix
            :param A: alpha list
            :param device: cpu or gpu
            """
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")

        # original gpt2 model
        input_ids = input_ids.to(device)
        outputs = transformer(input_ids=input_ids)[0][0][-1]  # (2, batch, len, dim)
        outputs = outputs.cpu().detach().numpy()
        logits = embedding.dot(outputs)

        old_rank = np.argsort(-logits).tolist()
        old_logits = np.sort(-logits).tolist()
        topk_raw = old_rank[:k]
        logits_raw = [-x for x in old_logits[:k]]

        # averaged hidden state debiased gpt2 model
        outputs_P = P.dot(outputs.T).T
        KL1 = [0 for ii in range(len(A))]
        KL2 = [0 for ii in range(len(A))]
        for a in range(len(A)):
            avg_outputs = A[a] * outputs + (1 - A[a]) * outputs_P
            avg_logits = embedding.dot(avg_outputs)

            logits_new = []
            for i, token in enumerate(topk_raw):
                logits_new.append(avg_logits[token])
            logits_new = np.array(logits_new)

            KL1[a] = scipy.stats.entropy(logits_raw, logits_new)
            KL2[a] = scipy.stats.entropy(logits_new, logits_raw)

        return KL1 + KL2


    def topk_kl_overlap_subspace(self, prompt_text, k, tokenizer, model, embedding, transformer, mode, device):
        if mode[1] == "gender":
            if mode[0] == "direction":
                gender_direction = np.load(sys.path[1]+"data/bias_subspace/gpt2_gender_direction.npy")
                debiased_embedding = np.array([self.drop(embedding[i], gender_direction) for i in range(embedding.shape[0])])
            else:
                gender_direction = np.load(sys.path[1]+"data/bias_subspace/gpt2_gender_subspace.npy")
                debiased_embedding = np.array([self.dropspace(embedding[i], gender_direction) for i in range(embedding.shape[0])])
        else:
            religion_dir1 = np.load(sys.path[1]+"data/bias_subspace/religion_direction1.npy")
            religion_dir2 = np.load(sys.path[1]+"data/bias_subspace/religion_direction2.npy")
            religion_dir3 = np.load(sys.path[1]+"data/bias_subspace/religion_direction3.npy")
            debiased_embedding = np.array([self.drop(embedding[i], religion_dir1) for i in range(embedding.shape[0])])
            debiased_embedding = np.array([self.drop(debiased_embedding[i], religion_dir2) for i in range(embedding.shape[0])])
            debiased_embedding = np.array([self.drop(debiased_embedding[i], religion_dir3) for i in range(embedding.shape[0])])

        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")

        # original gpt2 model
        input_ids = input_ids.to(device)
        outputs = transformer(input_ids=input_ids)[0][0][-1]  # (2, batch, len, dim)
        outputs = outputs.cpu().detach().numpy()
        logits = embedding.dot(outputs)

        old_rank = np.argsort(-logits).tolist()
        old_logits = np.sort(-logits).tolist()
        topk_raw = old_rank[:k]
        logits_raw = [-x for x in old_logits[:k]]

        # averaged hidden state debiased gpt2 model
        avg_outputs = outputs
        avg_logits = debiased_embedding.dot(avg_outputs)

        logits_new = []
        for i, token in enumerate(topk_raw):
            logits_new.append(avg_logits[token])
        logits_new = np.array(logits_new)

        KL1 = scipy.stats.entropy(logits_raw, logits_new)
        KL2 = scipy.stats.entropy(logits_new, logits_raw)

        return (KL1 + KL2) / 2
    def local_kl(self, male_context, female_context, tokenizer, model, embedding, transformer, P, A, device):
        kl1_avg = [0. for ii in range(len(A))]
        kl2_avg = [0. for ii in range(len(A))]
        for i in range(male_context.shape[0]):
            input_ids_m = tokenizer.encode(male_context[i], add_special_tokens=False, return_tensors="pt")
            input_ids_m = input_ids_m.to(device)
            outputs = transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
            outputs_P = P.dot(outputs.T).T

            input_ids_f = tokenizer.encode(female_context[i], add_special_tokens=False, return_tensors="pt")
            input_ids_f = input_ids_f.to(device)
            outputs_f = transformer(input_ids=input_ids_f)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
            outputs_P_f = P.dot(outputs_f.T).T

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

                KL1 = scipy.stats.entropy(probs_m[0].detach().numpy(), probs_f[0].detach().numpy())
                KL2 = scipy.stats.entropy(probs_f[0].detach().numpy(), probs_m[0].detach().numpy())

                kl1_avg[a] += KL1
                kl2_avg[a] += KL2

        return kl1_avg, kl2_avg
    def local_kl_reverse(self, occ_context, tokenizer, model, embedding, transformer, pairs_id, A, P, device):
        kl = [0. for ii in range(len(A))]
        for i in range(occ_context.shape[0]):
            input_ids = tokenizer.encode(occ_context[i], add_special_tokens=False, return_tensors="pt")
            input_ids = input_ids.to(device)
            outputs = transformer(input_ids=input_ids)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
            outputs_P = P.dot(outputs.T).T

            for a in range(len(A)):
                outputs_P = (1 - A[a]) * outputs_P + A[a] * outputs
                new_logits = embedding.dot(outputs_P)
                new_logits = torch.from_numpy(new_logits).float()
                new_logits = new_logits.unsqueeze(0)  # [1, 50257]
                probs = F.softmax(new_logits, dim=-1)
                probs = probs.cpu().detach().numpy()

                occ_prob1 = 0.
                occ_prob2 = 0.
                for p1, p2 in pairs_id:
                    occ_prob1 += probs[0][p1]
                    occ_prob2 += probs[0][p2]

                tmp_kl1 = 0.
                tmp_kl2 = 0.
                for p1, p2 in pairs_id:
                    tmp_kl1 += probs[0][p1]/occ_prob1 * np.log(probs[0][p1] / occ_prob1 / probs[0][p2] * occ_prob2)
                    tmp_kl2 += probs[0][p2]/occ_prob2 * np.log(probs[0][p2] / occ_prob2 / probs[0][p1] * occ_prob1)
                kl[a] += (tmp_kl1 + tmp_kl2) / 2

        return kl


    def local_kl_reverse_geometry(self, occ_context, tokenizer, model, embedding, transformer, pairs_id, num_components=2, device="cpu"):
        def doPCA(pairs, num_components=10):
            matrix = []
            for a, b in pairs:
                center = (a + b)/2
                norm_a = a - center
                norm_b = b - center
                norm_a, norm_b = norm_a.detach().numpy(), norm_b.detach().numpy()
                # norm_a, norm_b = norm_a/np.linalg.norm(norm_a), norm_b/np.linalg.norm(norm_b)
                matrix.append(norm_a)
                matrix.append(norm_b)
            matrix = np.array(matrix)
            pca = PCA(n_components=num_components, svd_solver="full")
            pca.fit(matrix) # Produce different results each time...
            return pca

        def dropspace(u, V):
            # u, V = u.detach().numpy(), V.detach().numpy()
            norm_sqrd = np.sum(V*V, axis=-1)
            vecs = np.divide(V@u, norm_sqrd)[:, None] * V
            subspace = np.sum(vecs, axis=0)
            return u - subspace

        pairs = []
        for female, male in pairs_id:
            female_feat, male_feat = embedding[female], embedding[male]
            female_feat, male_feat = female_feat/np.linalg.norm(female_feat), male_feat/np.linalg.norm(male_feat)
            if type(male_feat) is np.ndarray:
                female_feat, male_feat = torch.from_numpy(female_feat), torch.from_numpy(male_feat)
            pairs.append((female_feat, male_feat))
        pca_res = doPCA(pairs, num_components=num_components)
        print("pca_res.explained_variance_ratio_: ", pca_res.explained_variance_ratio_)
        print("pca shape", pca_res.components_.shape)
        gender_dir1 = torch.from_numpy(pca_res.components_[0])
        gender_dir2 = torch.from_numpy(pca_res.components_[1])
        # gender_dir = torch.from_numpy(pca_res.components_[:num_components])
        gender_dir = pca_res.components_[:num_components]

        # kl = [0. for ii in range(len(A))]
        kl = 0.
        for i in range(occ_context.shape[0]):
            input_ids = tokenizer.encode(occ_context[i], add_special_tokens=False, return_tensors="pt")
            input_ids = input_ids.to(device)
            outputs = transformer(input_ids=input_ids)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
            logits = embedding.dot(outputs)
            logits = torch.from_numpy(logits).float()
            logits = logits.unsqueeze(0)
            probs = F.softmax(logits, dim=-1)
            probs = probs.cpu().detach().numpy()

            occ_prob1 = 0.
            occ_prob2 = 0.
            for p1, p2 in pairs_id:
                occ_prob1 += probs[0][p1]
                occ_prob2 += probs[0][p2]

            tmp_kl1 = 0.
            tmp_kl2 = 0.
            for p1, p2 in pairs_id:
                tmp_kl1 += probs[0][p1] / occ_prob1 * np.log(probs[0][p1] / occ_prob1 / probs[0][p2] * occ_prob2)
                tmp_kl2 += probs[0][p2] / occ_prob2 * np.log(probs[0][p2] / occ_prob2 / probs[0][p1] * occ_prob1)
            kl += (tmp_kl1 + tmp_kl2) / 2

        tmp = model.lm_head.weight.data
        model.lm_head.weight.data = torch.from_numpy(
            np.array([dropspace(embedding[i], gender_dir) for i in range(embedding.shape[0])]))

        kl_debias = 0.
        for i in range(occ_context.shape[0]):
            input_ids = tokenizer.encode(occ_context[i], add_special_tokens=False, return_tensors="pt")
            input_ids = input_ids.to(device)
            outputs = transformer(input_ids=input_ids)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
            logits = embedding.dot(outputs)
            logits = torch.from_numpy(logits).float()
            logits = logits.unsqueeze(0)
            probs = F.softmax(logits, dim=-1)
            probs = probs.cpu().detach().numpy()

            occ_prob1 = 0.
            occ_prob2 = 0.
            for p1, p2 in pairs_id:
                occ_prob1 += probs[0][p1]
                occ_prob2 += probs[0][p2]

            tmp_kl1 = 0.
            tmp_kl2 = 0.
            for p1, p2 in pairs_id:
                tmp_kl1 += probs[0][p1] / occ_prob1 * np.log(probs[0][p1] / occ_prob1 / probs[0][p2] * occ_prob2)
                tmp_kl2 += probs[0][p2] / occ_prob2 * np.log(probs[0][p2] / occ_prob2 / probs[0][p1] * occ_prob1)
            kl_debias += (tmp_kl1 + tmp_kl2) / 2

            # outputs_P = P.dot(outputs.T).T

            # for a in range(len(A)):
            #     outputs_P = (1 - A[a]) * outputs_P + A[a] * outputs
            #     new_logits = embedding.dot(outputs_P)
            #     new_logits = torch.from_numpy(new_logits).float()
            #     new_logits = new_logits.unsqueeze(0)  # [1, 50257]
            #     probs = F.softmax(new_logits, dim=-1)
            #     probs = probs.cpu().detach().numpy()

        #model.lm_head.weight.data = tmp

        return kl, kl_debias