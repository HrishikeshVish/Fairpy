import sys
from BiasDetection.utils.configuration import configuration
import numpy as np
import torch
import argparse
from BiasDetection.utils.bias_utils import collate, how_many_tokens, find_mask_token, extract_gendered_profession_emb, get_vader_score, get_gendered_profs
import pandas as pd
#from model import Aligned_BERT
from tqdm import tqdm
from pattern3.en import pluralize, singularize
from copy import copy
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import regex as re
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

import glob

class LogProbabilityNationality():
    def __init__(self):
        self.nationality = configuration['en']['nationality']


        self.nationality = configuration['en']['nationality']
        #MSK = configuration['en']['MSK']
        self.en_nationality = configuration['en']['nationality']
        self.occ_path = sys.path[1]+'data/'+ configuration['en']['occ_path']
        #occ_path = configuration['en']['occ_path']
        # Occupation Loading
        with open(self.occ_path, 'r') as f:
            tt = f.readlines()

        self.occ = []

        for i in range(len(tt)):
            self.occ.append(tt[i].rstrip())

        #print("Occupations loading complete!")

        # Loading Templates
        template_path = sys.path[1]+'data/'+configuration['en']['template_path']
        with open(template_path, 'r') as f:
            tt = f.readlines()

        self.saved_templates = []

        for i in range(len(tt)):
            self.saved_templates.append(tt[i].rstrip())
        #print("Templates loading complete!")

    def log_probability_for_single_sentence(self, model, tokenizer, device,MSK,
                                            template, attr, nation_dict, last=False, use_pretrained=False):

        col_dict = collate(self.en_nationality, self.nationality)
        vocab = tokenizer.get_vocab()
        softmax = torch.nn.Softmax(dim=0)
        results = []

        attribute_num = len(tokenizer.tokenize(attr))
        for number in nation_dict.keys():
            nations = nation_dict[number]
            how_many = int(number)
            #print("INT OF HOW MANY ", how_many)
            #print("Number = ", number)
            
            target_mask = ' '.join([MSK for _ in range(how_many)]) #One mask for countries with one word. 2 for countries with 2 words
            attribute_mask = ' '.join([MSK for _ in range(attribute_num)])
            

            if '[AAA]' in template:
                sentence = template.replace('[TTT]', target_mask).replace('[AAA]', attr)
                prior_sentence = template.replace('[TTT]', target_mask).replace('[AAA]', attribute_mask)
            else:
                sentence = template.replace('[TTT]', target_mask).replace('[AAAs]', pluralize(attr))
                prior_sentence = template.replace('[TTT]', target_mask).replace('[AAAs]', attribute_mask)
            
            input_ids = tokenizer(sentence, return_tensors='pt').to(device)
            
            if not use_pretrained:
                target_prob = model(**input_ids).to(device) #Generate the target
            else:
                target_prob = model(**input_ids)[0].to(device)

            prior_input_ids = tokenizer(prior_sentence, return_tensors='pt').to(device)

            if not use_pretrained:
                prior_prob = model(**prior_input_ids).to(device)
            else:
                prior_prob = model(**prior_input_ids)[0].to(device)
            
            masked_tokens = find_mask_token(tokenizer, sentence, how_many, MSK)
            #print("PRIOR_SENTENCE = ", prior_sentence)
            #print("MASK = ", MSK)
            masked_tokens_prior = find_mask_token(tokenizer, prior_sentence, how_many, MSK, last) #Find location of mask in encoded sentence
            logits = []
            prior_logits = []
            for mask in masked_tokens:                
                logits.append(softmax(target_prob[0][mask]).detach())

            for mask in masked_tokens_prior:
                prior_logits.append(softmax(prior_prob[0][mask]).detach())

            for nat in nations:

                ddf = [col_dict[nat]]
                nat_logit = 1.0
                nat_prior_logit = 1.0

                for token in tokenizer.tokenize(nat):

                    for logit in logits:
                        temp = float(logit[vocab[token]].item())
                        if(temp>0 or temp<0):
                            nat_logit *= float(logit[vocab[token]].item())
                            #print(logit[vocab[token]], token)
                    for prior_logit in prior_logits:
                        temp = float(prior_logit[vocab[token]].item())
                        if(temp > 0 or temp < 0):
                            nat_prior_logit *= float(prior_logit[vocab[token]].item())
                            #print(prior_logit[vocab[token]], token)
                
                #print("LOG = ", np.log(float(nat_logit/nat_prior_logit)))
                
                
                ddf.append(np.log(float(nat_logit / nat_prior_logit)))
                results.append(np.array(ddf))
        return pd.DataFrame(results, columns=['nationality', 'normalized_prob'], dtype=(float)).sort_values(
            "normalized_prob", ascending=False)


    def log_probability_for_single_sentence_multiple_attr(self, model, tokenizer, device, MSK,
                                                        template, occ, nation_dict, use_pretrained=False):
        last = False
        if template.find('[TTT]') > template.find('[AAA]') and template.find('[TTT]') > template.find('[AAAs]'):
            last = True

        mean_scores = []
        var_scores = []
        std_scores = []

        for attr in occ:
            ret_df = self.log_probability_for_single_sentence(model, tokenizer, device, MSK,
                                                        template, attr, nation_dict, last, use_pretrained)
            #print(attr)
            #print(ret_df)      
            mean_scores.append(ret_df['normalized_prob'].mean())
            var_scores.append(ret_df['normalized_prob'].var())
            std_scores.append(ret_df['normalized_prob'].std())

        mean_scores = np.array(mean_scores)
        var_scores = np.array(var_scores)
        std_scores = np.array(std_scores)

        return mean_scores, var_scores, std_scores


    def log_probability_for_multiple_sentence(self, model, tokenizer, device, MSK, templates=[], occ=[], use_pretrained=False):
        if(templates == []):
            templates = self.saved_templates
        if(occ == []):
            occ = self.occ
        nation_dict = how_many_tokens(self.nationality, tokenizer)
        total_mean = []
        total_var = []
        total_std = []

        for template in tqdm(templates):
            m, v, s = self.log_probability_for_single_sentence_multiple_attr(model, tokenizer, device, MSK,
                                                                        template, occ, nation_dict, use_pretrained)

            total_mean.append(m.mean())
            total_var.append(v.mean())
            total_std.append(s.mean())

        return total_mean, total_var, total_std

