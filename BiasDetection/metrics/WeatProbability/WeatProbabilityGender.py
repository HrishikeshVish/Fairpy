from BiasDetection.metrics.WeatProbability import WeatProbability
import sys
import numpy as np
import os
import time
class WeatProbabilityGender(WeatProbability.WeatProbability):
    def __init__(self,file_write=False,output_dir=sys.path[1]+'res/local_res/'):
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
        return
    def probabiliy_of_real_next_token(self, model, tokenizer, embedding, device, transformer, P=None, A=None):
        if(P == None):
            P = self.P
        if(A == None):
            A = self.A
        ### Local Metric2.2: Weat_true_label - reflect language model
        t1 = time.time()
        print('-'*100, file=self.f)
        print("### Local Metric2.2: Weat_true_label - reflect language model ###", file=self.f)

        weat_corpus = np.loadtxt(sys.path[1]+"data/weat_corpus.txt", dtype=str, delimiter="\n")[:30]

        weat_dataset = []
        weat_pos = []
        for sentence in weat_corpus:
            input_ids = tokenizer.encode(sentence, add_special_tokens=False, return_tensors="pt")
            next_token_id = input_ids[0][-1]
            input_ids = input_ids[:, :-1]

            weat_dataset.append((sentence, input_ids))
            weat_pos.append(next_token_id)

        # avg debias
        res_true_label = super().weat_true_label(weat_dataset, weat_pos, model, embedding, transformer, A, P, self.p, device, topk=False)
        print("average: ", res_true_label, file=self.f)

        res_true_label_subspace = super().weat_true_label_subspace(weat_dataset, weat_pos, model, embedding, ["direction", "gender", "token"],transformer, self.p, device, A, P, topk=False)
        print("subspace: ", res_true_label_subspace, file=self.f)
        return res_true_label, res_true_label_subspace
