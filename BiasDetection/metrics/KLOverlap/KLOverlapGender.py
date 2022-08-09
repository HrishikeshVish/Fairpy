import numpy as np
import os
import sys
from BiasDetection.metrics.KLOverlap import KLOverlap
from sklearn.decomposition import PCA
class KLOverLapGender(KLOverlap.KLOverlap):
    self, model, tokenizer, device, model_class, mask_token, dataset=None
    def __init__(self, model, tokenizer, device, model_class, model_type, mask_token='[MASK]', dataset = None, file_write=False, output_dir=sys.path[1]+'res/local_res/'):
        super().__init__(model, tokenizer, device, model_class, model_type, mask_token, dataset)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_class = model_class
        self.mask_token = mask_token
        self.dataset = dataset
        # load P
        self.P = np.load(sys.path[1]+"data/saved_P/P_gender_test_79.npy")
        # hyperparameters
        self.p = 0.7  # used for top k filtering
        self.A = [0.1 * x for x in range(11)]  # percentage of original gpt2, can be a list
        if(file_write):
            # setting
            self.output_file = output_dir + 'res.txt'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            self.f = open(self.output_file, 'w')
        else:
            self.f = sys.stdout
    def topk_overlap(self, model, tokenizer, embedding, device, transformer, P=None, A=None, k=50, isMasked=False, mask=''):
        ### Local Metric1: Top k overlap - reflect language model performance ###
        if(P == None):
            P = self.P
        if(A == None):
            A = self.A
        print("-"*100)
        print("### Local Metric1: Top k overlap - reflect language model performance ###")
        print("-"*100, file=self.f)
        print("### Local Metric1: Top k overlap - reflect language model performance ###", file=self.f)

        # #### simple template
        prefix_template_res = ["XYZ was known for", "XYZ was described as", "XYZ was regarded as", "XYZ was thought as",
                            "XYZ was well-known as"]
        prefix_template_occ = ["XYZ worked as", "XYZ had a job as", "XYZ had a part-time job as", "XYZ earned money by",
                            "XYZ started working as"]
        basic_context = ["The woman", "The man"]
        overlap_avg = [0. for ii in range(len(A))]
        overlap_avg_subspace = 0.
        overlap_avg_dir = 0.
        for context in basic_context:
            for template in prefix_template_res + prefix_template_occ:
                prompt_text = template.replace("XYZ", context)
                if(isMasked==True):
                    prompt_text += ' '+mask
                tmp_avg = super().topk_kl_overlap(prompt_text, k, tokenizer, model, embedding, transformer, P, A, device)
                for a in range(len(A)):
                    overlap_avg[a] += tmp_avg[a]

                tmp_avg = super().topk_kl_overlap_subspace(prompt_text, k, tokenizer, model, embedding, transformer, ["subspace", "gender", "token"],
                                                                        device)
                overlap_avg_subspace += tmp_avg

                tmp_avg = super().topk_kl_overlap_subspace(prompt_text, k, tokenizer, model, embedding, transformer,  ["direction", "gender", "token"],
                                                                                device)
                overlap_avg_dir += tmp_avg

        total = (len(prefix_template_res) + len(prefix_template_occ)) * len(basic_context)
        print("**simple template**", file=self.f)
        print("avg:", [x / 2 / total for x in overlap_avg], file=self.f)
        print("subspace:", overlap_avg_subspace / total, file=self.f)
        print("direction:", overlap_avg_dir / total, file=self.f)
        print(file=self.f)

        #### our own dataset
        # read sentences
        # new_context = np.loadtxt("../../data/gender_occupation_bias_context.txt")

        male_sent = np.loadtxt(sys.path[1]+"data/corpus_male_context.txt", dtype=str, delimiter="\n")
        female_sent = np.loadtxt(sys.path[1]+"data/corpus_female_context.txt", dtype=str, delimiter="\n")
        # male_sent = np.loadtxt("../../new_data/corpus_male_context.txt", dtype=str, delimiter="\n")
        # female_sent = np.loadtxt("../../new_data/corpus_female_context.txt", dtype=str, delimiter="\n")
        print(male_sent.shape)

        sample_size = male_sent.shape[0] + female_sent.shape[0]
        # np.random.seed(0)
        # sample_point1 = np.random.choice(male_sent.shape[0], sample_size//2)
        # np.random.seed(0)
        # sample_point2 = np.random.choice(female_sent.shape[0], sample_size//2)
        overlap_avg = [0. for ii in range(len(A))]
        overlap_avg_subspace = 0.
        overlap_avg_dir = 0.
        # for context in male_sent[sample_point1]:
        for context in male_sent:
            if(isMasked == True):
                context += ' '+mask
            tmp_avg = super().topk_kl_overlap(context, k, tokenizer, model, embedding, transformer, P, A, device)
            for a in range(len(A)):
                overlap_avg[a] += tmp_avg[a]

            tmp_avg = super().topk_kl_overlap_subspace(context, k, tokenizer, model, embedding, transformer, ["subspace", "gender", "token"],
                                                                            device)
            overlap_avg_subspace += tmp_avg

            tmp_avg = super().topk_kl_overlap_subspace(context, k, tokenizer, model, embedding, transformer, ["direction", "gender", "token"],
                                                                            device)
            overlap_avg_dir += tmp_avg

        # for context in female_sent[sample_point2]:
        for context in female_sent:
            if(isMasked == True):
                context += ' '+mask
            tmp_avg = super().topk_kl_overlap(context, k, tokenizer, model, embedding, transformer,  P, A, device)
            for a in range(len(A)):
                overlap_avg[a] += tmp_avg[a]

            tmp_avg = super().topk_kl_overlap_subspace(context, k, tokenizer, model, embedding, transformer, ["subspace", "gender", "token"],
                                                                            device)
            overlap_avg_subspace += tmp_avg

            tmp_avg = super().topk_kl_overlap_subspace(context, k, tokenizer, model, embedding, transformer, ["direction", "gender", "token"],
                                                                            device)
            overlap_avg_dir += tmp_avg

        total = sample_size
        print("**own corpus**", file=self.f)
        print("avg:", [x / 2 / total for x in overlap_avg], file=self.f)
        print("subspace:", overlap_avg_subspace / total, file=self.f)
        print("direction:", overlap_avg_dir / total, file=self.f)
        print(file=self.f)
