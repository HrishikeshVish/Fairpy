from BiasMaskedLM.configuration import configuration
import numpy as np
import torch
import argparse
from BiasMaskedLM.bias_utils import collate, how_many_tokens, find_mask_token
import pandas as pd
#from model import Aligned_BERT
from tqdm import tqdm
"""
parser = argparse.ArgumentParser()
parser.add_argument('--language', type=str, required=False, default='en')
parser.add_argument('--custom_model_path', type=str, default=None)
parser.add_argument('--num', default=0)


args = parser.parse_args()

use_pretrained = True if args.custom_model_path is None else False



bert_model = configuration[args.language]['bert_model']





en_nationality = configuration['en']['nationality']

#tokenizer = BertTokenizer.from_pretrained(bert_model)
#tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#tokenizer = T5Tokenizer.from_pretrained('t5-small')
#tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
#model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

#tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

#model = RobertaForMaskedLM.from_pretrained("roberta-base")

tokenizer = TapasTokenizer.from_pretrained("google/tapas-base")

model = TapasForMaskedLM.from_pretrained("google/tapas-base")

MSK = tokenizer.encode('[MASK]')[0]
#tokenizer.add_special_tokens({'additional_special_tokens':['[MASK]']})
#MSK = tokenizer.encode('[MASK]')[0]

config = AutoConfig.from_pretrained(bert_model)
#config = AutoConfig.from_pretrained('gpt2')
#config = AutoConfig.from_pretrained('t5-small')
device = torch.device("cuda:"+str(args.num))
#device = torch.device('cpu')
if args.custom_model_path:
    print("Model Loading!")
    model = torch.load(args.custom_model_path, map_location=device)
else:
    print("Using pretrained model!")
    #model = BertForMaskedLM.from_pretrained(bert_model)
    #model = GPT2LMHeadModel.from_pretrained('gpt2')
    #model.resize_token_embeddings(len(tokenizer))
    #model = T5ForConditionalGeneration.from_pretrained('t5-small')

if args.language == 'en':
    from pattern3.en import pluralize, singularize
elif args.language == 'de':
    from pattern3.de import pluralize, singularize
elif args.language == 'es':
    from pattern3.es import pluralize, singularize
else:
    pass


model.eval()
model.to(device)
"""
from pattern3.en import pluralize, singularize
nationality = configuration['en']['nationality']
#MSK = configuration['en']['MSK']
en_nationality = configuration['en']['nationality']
occ_path = 'BiasMaskedLM/'+ configuration['en']['occ_path']
#occ_path = configuration['en']['occ_path']
# Occupation Loading
with open(occ_path, 'r') as f:
    tt = f.readlines()

occ = []

for i in range(len(tt)):
    occ.append(tt[i].rstrip())

print("Occupations loading complete!")

# Loading Templates
template_path = 'BiasMaskedLM/'+configuration['en']['template_path']
with open(template_path, 'r') as f:
    tt = f.readlines()

saved_templates = []

for i in range(len(tt)):
    saved_templates.append(tt[i].rstrip())
print("Templates loading complete!")

def log_probability_for_single_sentence(model, tokenizer, device,MSK,
                                        template, attr, nation_dict, last=False, use_pretrained=False):

    col_dict = collate(en_nationality, nationality)
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


def log_probability_for_single_sentence_multiple_attr(model, tokenizer, device, MSK,
                                                      template, occ, nation_dict, use_pretrained=False):
    last = False
    if template.find('[TTT]') > template.find('[AAA]') and template.find('[TTT]') > template.find('[AAAs]'):
        last = True

    mean_scores = []
    var_scores = []
    std_scores = []

    for attr in occ:
        ret_df = log_probability_for_single_sentence(model, tokenizer, device, MSK,
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


def log_probability_for_multiple_sentence(model, tokenizer, device, MSK, templates=saved_templates, occ=occ, use_pretrained=False):

    nation_dict = how_many_tokens(nationality, tokenizer)
    total_mean = []
    total_var = []
    total_std = []

    for template in tqdm(templates):
        m, v, s = log_probability_for_single_sentence_multiple_attr(model, tokenizer, device, MSK,
                                                                    template, occ, nation_dict, use_pretrained)

        total_mean.append(m.mean())
        total_var.append(v.mean())
        total_std.append(s.mean())

    return total_mean, total_var, total_std

"""total_mean, total_var, total_std = log_probability_for_multiple_sentence(model, tokenizer, templates, occ, use_pretrained=use_pretrained)

if use_pretrained:
    print("CB score of {} in {} : {}".format(bert_model, args.language, np.array(total_var).mean()))
else:
    print("CB score of {} (from weights {}) in {}: {}".format(bert_model, args.custom_model_path, args.language, np.array(total_var).mean()))
"""