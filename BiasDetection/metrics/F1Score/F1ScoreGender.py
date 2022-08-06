import sys
import pandas as pd
from BiasDetection.utils.bias_utils import get_vader_score, get_gendered_profs, f1_score, accuracy_score, extract_gendered_profession_emb
import regex as re
import torch
import numpy as np
from copy import copy
class F1ScoreGender():
    def __init__(self):
        return
    
    def data_formatter(self, filename, embed_data = False, mask_token = '[MASK]', model = None, tokenizer = None, baseline_tester= False, reverse = True, 
                        female_name = 'Alice', male_name = 'Bob'):
        """
        Formats data by masking pronoun and masked sentences in new file
        filename      - input WinoBias file
        embed_data    - if False:  Returns pro- and anti-stereotypical pronouns, the profession the pronoun refers to and the sentiment of sentences
                        if True: this function returns the final BERT embeddings of the profession token (needed for PCA)
        baseline_tester - 0 use WinoBias set
                        1 replace both professions by stereotypical names (used for testing baseline coreference performance)
                        2 replace referenced profession by stereotypical name
        reverse          - if baseline_tester is on, include sentences where names and pronouns are swapped 
                        e.g. for "Alice sees Bob and [she] asks...", also include "Bob sees Alice and [he] asks ... ". Decreases variance.
        mask_token    - mask token used by BERT model (either [MASK]  or <mask>)
        model         - specific BERT model
        tokenizer     - tokenizer used by BERT model
        """
        

        prodev1 = sys.path[1]+"data/WinoBias/wino/data/pro_stereotyped_type1.txt.dev"
        prodev2 = sys.path[1]+"data/WinoBias/wino/data/pro_stereotyped_type2.txt.dev"
        antidev1 = sys.path[1]+"data/WinoBias/wino/data/anti_stereotyped_type1.txt.dev"
        antidev2 = sys.path[1]+"data/WinoBias/wino/data/anti_stereotyped_type2.txt.dev"

        protest1 = sys.path[1]+"data/WinoBias/wino/data/pro_stereotyped_type1.txt.test"
        protest2 = sys.path[1]+"data/WinoBias/wino/data/pro_stereotyped_type2.txt.test"
        antitest1 = sys.path[1]+"data/WinoBias/wino/data/anti_stereotyped_type1.txt.test"
        antitest2 = sys.path[1]+"data/WinoBias/wino/data/anti_stereotyped_type2.txt.test"
        # Initialise
        masklabels = []
        professions = []
        sentiments = []

        # Experimenting with masking the he/she/his/her
        f = open(eval('pro'+filename), "r") 
        lines = f.readlines()
        f.close()
        f = open(eval('anti'+filename), "r") 
        lines_anti = f.readlines()
        f.close()
        if baseline_tester: mprofs, fprofs = get_gendered_profs()

        textfile = open(filename+'.txt', 'w')
        embedded_data = []
        for i,line in enumerate(lines):

            #chech if one of the words in the sentence is he/she/his/her
            mask_regex = r"(\[he\]|\[she\]|\[him\]|\[his\]|\[her\]|\[He\]|\[She\]|\[His\]|\[Her\])"
            pronoun = re.findall(mask_regex, line)
            if len(pronoun) == 1: ######## Dan/Dave what's the idea of this again?
                pronoun = pronoun[0][1:-1]
                pronoun_anti = re.findall(mask_regex, lines_anti[i])[0][1:-1]
        
                # Remove number at start of line
                new_line = re.sub(r"^(\d*)", "", line)
                new_line = re.sub(r"(.)$", " . ", new_line[1:])
        
        
                profession_pre = re.findall('\[(.*?)\]',new_line)[0]
                if profession_pre[1:4] == 'he ': 
                    profession = profession_pre[4:] # i.e. the/The
                elif profession_pre[0:2] =='a ':
                    profession = profession_pre[2:]
                else:
                    profession = profession_pre
                professions.append(profession)

                if embed_data:
                    try:
                        male_representation, female_representation, token_index, profession = extract_gendered_profession_emb(new_line, model, tokenizer)
                    # removes all square brackets
                    except:
                        continue
                new_line = re.sub(mask_regex, mask_token, new_line)
        
        
                new_line = re.sub(r'\[(.*?)\]',lambda L: L.group(1).rsplit('|', 1)[-1], new_line)
        
                # replace square brackets on MASK
                new_line = re.sub('MASK', '[MASK]', new_line)
        
                # Sentiment analysis of sentences
                sentiments.append([get_vader_score(line),get_vader_score(lines_anti[i]),get_vader_score(new_line)])
        
                if reverse:
                    new_line_rev = copy(new_line)

                if baseline_tester:
                    if pronoun in ('she', 'her'):
                        new_line = new_line.replace(profession_pre, female_name)
            
                else:
                    new_line = new_line.replace(profession_pre, male_name)
                if baseline_tester==1:
                    for prof in mprofs:
                        new_line = new_line.replace('The '+prof, male_name)
                        new_line = new_line.replace('the '+prof, male_name)
                        new_line = new_line.replace('a '+prof, male_name)
                        new_line = new_line.replace('A '+prof, male_name)
                
                    for prof in fprofs:
                        new_line = new_line.replace('The '+prof, female_name)
                        new_line = new_line.replace('the '+prof, female_name)
                        new_line = new_line.replace('a '+prof, female_name)
                        new_line = new_line.replace('A '+prof, female_name)

                new_line = new_line.lstrip().rstrip()
                textfile.write(new_line+ '\n')
                masklabels.append([pronoun,pronoun_anti])

                if reverse and baseline_tester:
                    if pronoun in ('she', 'her'):
                        new_line_rev = new_line_rev.replace(profession_pre, male_name)
            
                    else:
                        new_line_rev = new_line_rev.replace(profession_pre, female_name)
                    if baseline_tester==2:
                        for prof in fprofs:
                            new_line_rev = new_line_rev.replace('The '+prof, male_name)
                            new_line_rev = new_line_rev.replace('the '+prof, male_name)
                            new_line_rev = new_line_rev.replace('a '+prof, male_name)
                            new_line_rev = new_line_rev.replace('A '+prof, male_name)
                        for prof in mprofs:
                            new_line_rev = new_line_rev.replace('The '+prof, female_name)
                            new_line_rev = new_line_rev.replace('the '+prof, female_name)
                            new_line_rev = new_line_rev.replace('a '+prof, female_name)
                            new_line_rev = new_line_rev.replace('A '+prof, female_name)

                    textfile.write(new_line_rev)
                    masklabels.append([pronoun_anti,pronoun])
                    professions.append('removed prof')
                    sentiments.append([-100,-100,-100])
            
                if embed_data:
                    stereotypical_gender = pronoun.lower() not in ('she', 'her')
                    embedded_data.append([i, male_representation, female_representation, stereotypical_gender, profession, token_index])

                # write this line to new "masked" text file
        
                # print(line)
                # get the label without square brackets
                # print(new_m)
            else:
                pass

        #print(maskprodev1labels)
        textfile.close()
        # check it worked
        #f = open("maskprodev1.txt", "r") 
        #print(f.read())
        f.close()

        if embed_data:
            return embedded_data
        else:
            return masklabels, professions, np.array(sentiments)

    def predict(self, dataset, labels, professions, model, tokenizer, device, mask_token, use_elmo = 0, verbose= False, online_skew_mit = 0, female_name='Alice', male_name='Bob'):
        """
        Input:
        dataset             - dataset name (reads from .txt)
        labels              - possible pronouns (every entry contains stereotypical and anti-stereotypical option)
        professions         - professions that the pronoun references to
        use_elmo            - boolean that denotes to use ELMo or not
        verbose             - print wrong predictions
        online_skew_mit - 0 use BERT output pronoun ({him, his, he} vs {she, her} probabilities
                                1 divide default output by pronoun probabilities of sentences in which all professions are masked
                                2 divide default output by gender probabilities in which just the referenced profession is masked
        Output:
        df_output           - pandas dataframe with predictions, pro and anti-stereo pronouns, professions, probabilities for either gendered pronouns
        n_misk              - list with number of classifications for each gender
        n_misk_profs        - dictionary with number of classifications for each gender for each profession
        """
    
        predicted_output = []

        # read text file
        f = open(dataset+'.txt', "r") 
        lines = f.readlines()
        f.close()
        n_misk = [0,0]
        n_misk_prof = {}
        #if use_elmo: embedder_ELMo = load_elmo()

        for prof in set(professions):
            n_misk_prof[prof] = [0,0] # mistakes per profession
        # loop over lines
        print('Running on', len(lines), 'examples')
        mprofs,fprofs = get_gendered_profs()
        for idx,line in enumerate(lines):
        
            line_output = []
            # read the line and its label
            line = lines[idx]
            label = labels[idx][0]
            label_anti = labels[idx][1]
        
            # identify relevant tokens to compare
        
            if label.lower() not in ('she','her'):
                male_label = label
                female_label = label_anti
                g_index = 1
            else:
                male_label = label_anti
                female_label = label
                g_index = 0
        
            # if which_bert == 'BERT' or which_bert == 'distilBERT':
            # comparison_labels = [male_label,female_label]
            # elif which_bert == 'Roberta':
            #   comparison_labels = ['Ġ'+male_label,'Ġ'+female_label]
            # elif which_bert == 'Albert':
            #   comparison_labels = ['▁'+male_label,'▁'+female_label]
        
            comparison_labels = [male_label,female_label]
            #comparison_labels = [label,label_anti]
        
            comparison_indices = tokenizer.convert_tokens_to_ids(comparison_labels)
        
        
            # tokenise the line
            if use_elmo==0:
                input_ids = torch.tensor(tokenizer.encode(line)).unsqueeze(0).cuda()  # Batch size 1
                masked_index = (input_ids == tokenizer.convert_tokens_to_ids([mask_token])[0]).nonzero()
        
        
                masked_index = masked_index[0,-1]
                if online_skew_mit:
                    new_line = line
                    if online_skew_mit==1:
                        for prof in mprofs+fprofs+[female_name, male_name]:
                            new_line = new_line.replace(prof, mask_token)#+str(int(round(random.random()*100)))+']')
                    else:
                        new_line = new_line.replace(professions[idx], mask_token )
                    input_ids_2 = torch.tensor(tokenizer.encode(new_line)).unsqueeze(0)  # Batch size 1
                    masked_index_2 = (input_ids_2 == tokenizer.convert_tokens_to_ids([mask_token])[0]).nonzero()
            
                    masked_index_2 = masked_index_2[0,-1] # choose last mask token in sentence, that corresponds to he she

                #input_ids = tokenizer(line, return_tensors='pt').to(device)
                #print(tokenizer.convert_ids_to_tokens(input_ids.squeeze()), masked_index, line) #for debuggig

                with torch.no_grad(): #necessary?
                    outputs = model(input_ids)
            
                    #print(tokenizer.convert_ids_to_tokens(input_ids[:,masked_index])) # for debugging: Check that masked index is indeed correctly defined
                    #print(outputs.shape)
                    prediction_scores = outputs[0]
                    #print(line)
                    #print(input_ids)
                    ##print(outputs[0].shape)
                    #print(masked_index)
                    
                    scores = prediction_scores[0, masked_index]
                    probs = torch.nn.functional.softmax(scores)
                    predicted_index = torch.argmax(scores)
                    if online_skew_mit:
                        outputs_2 = model(input_ids_2, masked_lm_labels=input_ids_2)
                        prediction_scores_2 = outputs_2[1]
                        scores_2 = prediction_scores_2[0, masked_index_2]
                        probs_2 = torch.nn.functional.softmax(scores_2)
            
        
                predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

                if online_skew_mit:
                    male_prob = probs[comparison_indices[0]]/probs_2[comparison_indices[0]]
                    female_prob = probs[comparison_indices[1]]/probs_2[comparison_indices[1]]
                else:
                    male_prob = probs[comparison_indices[0]]
                    female_prob = probs[comparison_indices[1]]

            #elif use_elmo == 1:
            #    male_prob, female_prob = ELMoprobs(line, male_label, female_label, embedder_ELMo)
            #    predicted_token = None
        
            #else: ### deprecated method of using BERT embedding distance for classification
            #    male_prob, female_prob = BERTembeddingdistances(line, male_label, female_label, model, tokenizer)
            #if which_bert == 'Roberta' or which_bert == 'Albert':
            #  predicted_token = predicted_token[:]
            male_prob = float(male_prob)
            female_prob = float(female_prob)
            # Append results to list
            line_output.append(idx)
            line_output.append(predicted_token)
            line_output.append(float(male_prob))
            line_output.append(float(female_prob))
            line_output.append(label)
            line_output.append(label_anti)
            line_output.append(professions[idx])
            #line_output.append(predicted_token==label)
        
            predicted_output.append(line_output)
        
        
            predicted_token = [male_label, female_label][male_prob<female_prob]
            mistake_made = g_index != bool((float(male_prob)>float(female_prob))) 
        
            n_misk[male_prob<female_prob]+=1

            n_misk_prof[professions[idx]][male_prob<female_prob]+=1
        

            if verbose:
                if mistake_made:  
                    print("\n\n---------- RESULT {} ---------- \n Original Sentence = {} \n Top [MASK] Prediction = {} \n Male Probability = {} \n Female Probability = {}\n Sentiment of masked sentence = {}".format(idx+1,line,predicted_token, line_output[2], line_output[3],sentiments[idx,2]))
                    print('Possible labels:', male_label, female_label)
    
        df_output = pd.DataFrame(predicted_output, columns = ['line', 'Top [MASK] Prediction', 'Male Probability', 'Female Probability', 'True Label', 'Anti Label', 'Profession'])
    
        return df_output, n_misk, n_misk_prof

    def f1_score_gender_profession(self, model, tokenizer, device, mask_token, model_class):
        results = []
        automated = False # set to true for all results, but for demo bit of overkill
        baseline_tester = False # Test baseline performance (Alice and Bob system, see Section 5.1 of report)

        if automated: # run for all out-of-the-box methods
            #which_berts = ['BERT', 'RoBERTa', 'DistilBERT']
            online_skew_mit_methods_to_use = ['','-O'] # normal method and online method (denoted by -O suffix)
            datasets = ['test1','test2']
    
        else: #manually select one model and settings
            #which_berts = ['BERT']
            online_skew_mit_methods_to_use = [''] # Do not use skew mitigation method
            datasets = ['test2']

        for which_bert in [model_class]:
            for online_skew_mit, online_skew_string in enumerate(online_skew_mit_methods_to_use):
                print('%%%%%%%%%%%%%%%%%%%%%%%', model_class + online_skew_string , '%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                results.append([which_bert+'-'+online_skew_string])
                for dataset in datasets:
                    print('####################### Dataset '+dataset+' #####################')
                    labels, professions, sentiments = self.data_formatter(dataset, mask_token = mask_token,  baseline_tester = baseline_tester, reverse = True)
        
                    df_pred, n_mist, n_misk_profs = self.predict(dataset,labels, professions, model, tokenizer,device, mask_token, verbose = False, online_skew_mit = online_skew_mit , use_elmo = 0)
        
                    df_pred['Sentiment'] = sentiments[:,2]
                    labels = df_pred['True Label'].str.contains("she|her") == False
        
        
                    #predicted = 2-df_pred['Top [MASK] Prediction'].str.contains("she|her")-df_pred['Top [MASK] Prediction'].str.contains("he|his|him") # 0 if female, 1 if male, 2 if neither
        
                    predicted_mf = df_pred['Male Probability'] > df_pred['Female Probability']
        
                    # print number of predictions per gender
                    print("number of male vs female predictions", n_mist[1],':',n_mist[0])

                    f1_pro = f1_score(labels,predicted_mf)*100
                    f1_ant = f1_score(labels==False, predicted_mf)*100
                    accuracy_pro = accuracy_score(labels, predicted_mf)*100
                    accuracy_ant = accuracy_score(labels==False, predicted_mf)*100
        
                    f1_pro_F = f1_score(labels==False,predicted_mf==False)*100
                    f1_ant_F = f1_score(labels, predicted_mf==False)*100


                    print('accuracy_pro = ', accuracy_pro)
                    print('accuracy_ant = ', accuracy_ant)
                    print('Delta acc =',accuracy_pro-accuracy_ant)
                    print('f1 pro M =',f1_pro)
                    print('f1 ant M =',f1_ant)
                    print('Delta M =',f1_pro-f1_ant)
                    print('f1 pro F =',f1_pro_F)
                    print('f1 ant F =',f1_ant_F)
                    print('Delta F =',f1_pro_F-f1_ant_F)
                    stereo = (abs(f1_pro-f1_ant)+abs(f1_pro_F-f1_ant_F))/2
                    skew = (abs(f1_pro-f1_pro_F)+abs(f1_ant-f1_ant_F))/2
                    results[-1] +=[round(f1_pro,1),round(f1_ant,1),round(f1_pro_F,1),round(f1_ant_F,1), round(stereo,1), round(skew,1)]
                    # prints the dictionary of professions with number of times 
                    #print('Female ratio of assignments per profession')
                    #for prof in n_misk_profs.keys():
                    #    print(prof, n_misk_profs[prof][1]/(n_misk_profs[prof][1]+n_misk_profs[prof][0]))
                    #print('Male ratio of assignments per profession')
                    #for prof in n_misk_profs.keys():
                    #    print(prof, n_misk_profs[prof][0]/(n_misk_profs[prof][1]+n_misk_profs[prof][0]))
        return results