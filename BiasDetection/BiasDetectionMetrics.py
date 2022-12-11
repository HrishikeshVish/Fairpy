from abc import ABC, abstractmethod
import sys
sys.path.insert(1, 'BiasDetection/')
from metrics.LogProbability.LogProbabilityNationality import LogProbabilityNationality
from metrics.LogProbability.LogProbabilityGender import LogProbabilityGender
from metrics.LogProbability.LogProbabilityRace import LogProbabilityRace
from metrics.LogProbability.LogProbabilityReligion import LogProbabilityReligion
from metrics.F1Score.F1ScoreGender import F1ScoreGender
from metrics.KLOverlap.KLOverlapGender import KLOverLapGender
from metrics.HellingerDistance.HellingerDistanceGender import HellingerDistanceGender
from metrics.WeatProbability.WeatProbabilityGender import WeatProbabilityGender
from metrics.StereoSetMetric.StereoSetGender import StereoSetGender
from metrics.StereoSetMetric.StereoSetRace import StereoSetRace
from metrics.StereoSetMetric.StereoSetProfession import StereoSetProfession
from metrics.StereoSetMetric.StereoSetReligion import StereoSetReligion
from metrics.StereoSetMetric.StereoSetOverall import StereoSetOverall
from metrics.Honest.HonestMetric import HonestMetric
from metrics.WeatScore import WeatScoreAge, WeatScoreGender, WeatScoreHealth, WeatScoreRace, WeatScoreReligion
from glob import glob
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer
from transformers import (
    CTRLLMHeadModel, CTRLTokenizer,
    GPT2LMHeadModel, GPT2Tokenizer,
    OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
    TransfoXLLMHeadModel, TransfoXLTokenizer,
    BertLMHeadModel,
    DistilBertModel,
    RobertaForCausalLM,
    AlbertModel,
    XLMTokenizer, XLMWithLMHeadModel,
    XLNetLMHeadModel, XLNetTokenizer,
    BertForMaskedLM, BertTokenizer,
    DistilBertForMaskedLM, DistilBertTokenizer,
    RobertaForMaskedLM, RobertaTokenizer,
    AlbertForMaskedLM, AlbertTokenizer,
)
class LMBiasDetection(ABC):
    def __init__(self, model_class, model_path, write_to_file, use_pretrained):
        self.use_pretrained = use_pretrained
        self.write_to_file = write_to_file
        self.model_class = model_class
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    @abstractmethod
    def load_model(self):
        pass

    

class CausalLMBiasDetection(LMBiasDetection):

    def __init__(self, model_class='',model_path='', write_to_file=False, use_pretrained=True, model = None, tokenizer = None):
        super().__init__(model_class, model_path, write_to_file, use_pretrained)
        self.PRE_TRAINED_MODEL_CLASS = {
        "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
        "gpt2-medium": (GPT2LMHeadModel, GPT2Tokenizer),
        "gpt2-large": (GPT2LMHeadModel, GPT2Tokenizer),
        "gpt2-xl": (GPT2LMHeadModel, GPT2Tokenizer),
        "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
        "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
        "xlnet-base-cased": (XLNetLMHeadModel, XLNetTokenizer),
        "xlnet-large-cased": (XLNetLMHeadModel, XLNetTokenizer),
        "transfo-xl-wt103": (TransfoXLLMHeadModel, TransfoXLTokenizer),
        "xlm-mlm-en-2048": (XLMWithLMHeadModel, XLMTokenizer),
        "bert-base-uncased": (BertLMHeadModel, BertTokenizer),
        "roberta-base": (RobertaForCausalLM, RobertaTokenizer),
        }
        self.MSK = '[MASK]'
        if('roberta' in model_class):
            self.MSK = '<mask>'
        self.config = ''
        if(use_pretrained == True):
            self.model, self.tokenizer = self.load_model(model_class, model_path, use_pretrained)
        else:
            self.model = model
            self.tokenizer = tokenizer
        #self.stereoSet = generativeBiasEval(self.model, self.device, tokenizer = self.tokenizer, input_file=sys.path[1]+'StereoSet/data/dev.json')
        if('xlnet' in model_class):
            self.embedding = self.model.lm_loss.weight.cpu().detach().numpy()
            self.transformer = self.model.transformer
        elif('xlm' in model_class):
            self.embedding = self.model.pred_layer.proj.weight.cpu().detach().numpy()
            self.transformer = self.model.transformer
        elif('transfo' in model_class):
            self.embedding = self.model.crit.out_layers[3].weight.cpu().detach().numpy()
            self.transformer = self.model.transformer
        elif('bert' not in model_class):
            self.embedding = self.model.lm_head.weight.cpu().detach().numpy()
            self.transformer = self.model.transformer
        else:
            if(model_class == 'bert-base-uncased'):
                self.embedding = self.model.cls.predictions.decoder.weight.cpu().detach().numpy()
                self.transformer = self.model.bert
            elif(model_class == 'roberta-base'):
                self.embedding = self.model.lm_head.decoder.weight.cpu().detach().numpy()
                self.transformer = self.model.roberta

    def load_model(self, model_class, model_path, use_pretrained):
        if(use_pretrained == False):
            use_pretrained = True # Need to figure out how to load custom tokenizers that work with custom models. 
        if(use_pretrained == True):
            if model_class not in self.PRE_TRAINED_MODEL_CLASS.keys():
                self.model_class = "gpt2"
                print("Specified model not Found. Using GPT-2 Instead")
            model, tokenizer = self.PRE_TRAINED_MODEL_CLASS[self.model_class]
            model = model.from_pretrained(model_class)
            tokenizer = tokenizer.from_pretrained(model_class)
            model = model.to(self.device)
        return model, tokenizer
    
    def topKOverlap(self, bias_type='gender', dataset=None, file_write=False, output_dir=''):
        if(bias_type == 'gender'):
            kl_overlap_obj = KLOverLapGender(self.model, self.tokenizer, self.device, self.model_class, 'causal',dataset=dataset, file_write=file_write, output_dir=output_dir)
            return kl_overlap_obj.evaluate(self.embedding, self.transformer)

    def hellingerDistance(self, bias_type='gender', dataset=None, file_write=False, output_dir=''):
        if(bias_type == 'gender'):
            hellinger_obj = HellingerDistanceGender(self.model, self.tokenizer, self.device, self.model_class, 'causal',self.MSK, dataset=None, file_write=file_write, output_dir=output_dir)
            return hellinger_obj.evaluate(self.embedding, self.transformer)
    def weatProbability(self, bias_type='gender', dataset=None, file_write=False, output_dir=''):
        if(bias_type == 'gender'):
            weat_prob_obj = WeatProbabilityGender(self.model, self.tokenizer, self.device, self.model_class, 'causal', self.MSK, dataset=dataset, file_write=file_write, output_dir=output_dir)
            return weat_prob_obj.evaluate(self.embedding, self.transformer)
    def stereoSetScore(self, bias_type='gender', dataset=None, metric_type='full'):
        if(bias_type == 'gender'):
            stereoset_obj = StereoSetGender(self.model, self.tokenizer, self.device, self.model_class, 'causal', self.MSK, dataset)
            return stereoset_obj.evaluate(metric_type)
        if(bias_type == 'race'):
            stereoset_obj = StereoSetRace(self.model, self.tokenizer, self.device, self.model_class, 'causal', self.MSK, dataset)
            return stereoset_obj.evaluate(metric_type)
        if(bias_type == 'religion'):
            stereoset_obj = StereoSetReligion(self.model, self.tokenizer, self.device, self.model_class, 'causal', self.MSK, dataset)
            return stereoset_obj.evaluate(metric_type)
        if(bias_type == 'profession'):
            stereoset_obj = StereoSetProfession(self.model, self.tokenizer, self.device, self.model_class, 'causal', self.MSK, dataset)
            return stereoset_obj.evaluate(metric_type)
        if(bias_type == 'all'):
            stereoset_obj = StereoSetOverall(self.model, self.tokenizer, self.device, self.model_class, 'causal', self.MSK, dataset)
            return stereoset_obj.evaluate(metric_type)

    def topKPercentage(self, bias_type='queer_nonqueer', dataset=None, k=5, lang='en', plot_graph=False):
        honest_obj = HonestMetric(self.model, self.tokenizer, self.device, self.model_class, 'causal', self.MSK, dataset, lang, bias_type, k)
        honest_score, honest_df = honest_obj.evaluate(plot_graph)
        return honest_score, honest_df
    def WeatScore(self, bias_type='gender', dataset=None):
        if(bias_type == 'gender'):
            weat_obj = WeatScoreGender.WeatScoreGender(self.model, self.tokenizer, self.device, self.model_class, 'causal', self.MSK, dataset)
            results = weat_obj.evaluate()
            p_score = 0
            e_score = 0
            counter = 0
            p_count = 0
            for result in results:
                if(result['p_value']<0.05):
                    p_score += result['p_value']
                    e_score += result['effect_size']
                    p_count+=1
            
                counter += 1
            
            #print(results)
            print('Percentage of p_value <0.05 ', p_count/counter)
            print('Average E-score ', e_score/p_count)
            return results
        if(bias_type == 'race'):
            weat_obj = WeatScoreRace.WeatScoreRace(self.model, self.tokenizer, self.device, self.model_class, 'causal', self.MSK, dataset)
            results = weat_obj.evaluate()
            p_score = 0
            e_score = 0
            counter = 0
            p_count = 0
            for result in results:
                if(result['p_value']<0.05):
                    p_score += result['p_value']
                    e_score += result['effect_size']
                    p_count+=1
            
                counter += 1
            
            #print(results)
            print('Percentage of p_value <0.05 ', p_count/counter)
            if(p_count>0):
                print('Average E-score ', e_score/p_count)
            else:
                print('Average E-score ', 0)
            return results
        if(bias_type == 'religion'):
            weat_obj = WeatScoreReligion.WeatScoreReligion(self.model, self.tokenizer, self.device, self.model_class, 'causal', self.MSK, dataset)
            results = weat_obj.evaluate()
            p_score = 0
            e_score = 0
            counter = 0
            p_count = 0
            for result in results:
                if(result['p_value']<0.05):
                    p_score += result['p_value']
                    e_score += result['effect_size']
                    p_count+=1
            
                counter += 1
            
            #print(results)
            print('Percentage of p_value <0.05 ', p_count/counter)
            if(p_count>0):
                print('Average E-score ', e_score/p_count)
            else:
                print('Average E-score ', 0)
            return results
        if(bias_type == 'age'):
            weat_obj = WeatScoreAge.WeatScoreAge(self.model, self.tokenizer, self.device, self.model_class, 'causal', self.MSK, dataset)
            results = weat_obj.evaluate()
            p_score = 0
            e_score = 0
            counter = 0
            p_count = 0
            for result in results:
                if(result['p_value']<0.05):
                    p_score += result['p_value']
                    e_score += result['effect_size']
                    p_count+=1
            
                counter += 1
            
            #print(results)
            print('Percentage of p_value <0.05 ', p_count/counter)
            if(p_count>0):
                print('Average E-score ', e_score/p_count)
            else:
                print('Average E-score ', 0)
            return results
        if(bias_type == 'health'):
            weat_obj = WeatScoreHealth.WeatScoreHealth(self.model, self.tokenizer, self.device, self.model_class, 'causal', self.MSK, dataset)
            results = weat_obj.evaluate()
            p_score = 0
            e_score = 0
            counter = 0
            p_count = 0
            for result in results:
                if(result['p_value']<0.05):
                    p_score += result['p_value']
                    e_score += result['effect_size']
                    p_count+=1
            
                counter += 1
            
            #print(results)
            print('Percentage of p_value <0.05 ', p_count/counter)
            if(p_count>0):
                print('Average E-score ', e_score/p_count)
            else:
                print('Average E-score ', 0)
            return results
        return
    def logProbability(self, bias_type='gender', dataset='crows', templates=None):
        if(bias_type == 'gender'):
            log_gender_obj = LogProbabilityGender(self.model, self.tokenizer, self.device, 'causal', self.model_class, self.MSK, dataset)
            return log_gender_obj.evaluate()
        elif(bias_type == 'race'):
            log_race_obj = LogProbabilityRace(self.model, self.tokenizer, self.device, 'causal', self.model_class, self.MSK, dataset)
            return log_race_obj.evaluate()
        elif(bias_type == 'religion'):
            log_religion_obj = LogProbabilityReligion(self.model, self.tokenizer, self.device, 'causal', self.model_class, self.MSK, dataset)
            return log_religion_obj.evaluate()
        

class MaskedLMBiasDetection(LMBiasDetection):
    def __init__(self, model_class='',model_path='', write_to_file=False, use_pretrained=True, model=None, tokenizer=None):
        super().__init__(model_class, model_path, write_to_file, use_pretrained)
        self.PRE_TRAINED_MODEL_CLASS = {
            "bert-base-uncased": (BertForMaskedLM, BertTokenizer),
            "bert-large-uncased": (BertForMaskedLM, BertTokenizer),
            "bert-base-cased": (BertForMaskedLM, BertTokenizer),
            "bert-large-cased": (BertForMaskedLM, BertTokenizer),
            "distilbert-base-uncased": (DistilBertForMaskedLM, DistilBertTokenizer),
            "distilbert-base-uncased-distilled-squad": (DistilBertForMaskedLM, DistilBertTokenizer),
            "distilroberta-base" : (DistilBertForMaskedLM, DistilBertTokenizer),
            "roberta-base": (RobertaForMaskedLM, RobertaTokenizer),
            "roberta-large": (RobertaForMaskedLM, RobertaTokenizer),
            "roberta-base-openai-detector": (RobertaForMaskedLM, RobertaTokenizer),
            "roberta-large-openai-detector": (RobertaForMaskedLM, RobertaTokenizer),
            "albert-base-v1": (AlbertForMaskedLM, AlbertTokenizer),
            }
        self.datasets = {
            'bec-Pro', 'winobias', 'custom-template'
        }
        if(use_pretrained == True):
            self.model, self.tokenizer = self.load_model(model_class, model_path, use_pretrained)
        else:
            self.model = model
            self.tokenizer = tokenizer
        #self.stereoSet = discriminativeBiasEval(self.model, self.device, tokenizer = self.tokenizer, input_file=sys.path[1]+'StereoSet/data/dev.json')
        self.config = ''
        self.MSK = '[MASK]'
        if('roberta' in model_class):
            self.MSK = '<mask>'
        
        if('bert' not in model_class):
            self.embedding = self.model.lm_head.weight.cpu().detach().numpy()
            self.transformer = self.model.transformer
        else:
            if(model_class == 'bert-base-uncased'):
                self.embedding = self.model.cls.predictions.decoder.weight.cpu().detach().numpy()
                self.transformer = self.model.bert
            elif(model_class == 'roberta-base'):
                self.embedding = self.model.lm_head.decoder.weight.cpu().detach().numpy()
                self.transformer = self.model.roberta
    
    def load_model(self, model_class, model_path, use_pretrained):
        if(use_pretrained == False):
            use_pretrained = True
        if(use_pretrained == True):
            if model_class not in self.PRE_TRAINED_MODEL_CLASS.keys():
                self.model_class = 'bert-base-uncased'
                print("Specified model not Found. Using BERT instead")
            model, tokenizer = self.PRE_TRAINED_MODEL_CLASS[self.model_class]
            model = model.from_pretrained(model_class)
            tokenizer = tokenizer.from_pretrained(model_class)
            model = model.to(self.device)
        return model, tokenizer
    def hellingerDistance(self, bias_type='gender', dataset=None, file_write=False, output_dir=''):
        if(bias_type == 'gender'):
            hellinger_obj = HellingerDistanceGender(self.model, self.tokenizer, self.device, self.model_class, 'masked',self.MSK, dataset=None, file_write=file_write, output_dir=output_dir)
            return hellinger_obj.evaluate(self.embedding, self.transformer)
    def logProbability(self, bias_type='gender', dataset='crows', templates=None):
        if(bias_type == 'nationality'):
            log_nationality_obj = LogProbabilityNationality(self.model, self.tokenizer, self.device, self.model_class, 'masked', '[MASK]', dataset)
            total_mean, total_var, total_std = log_nationality_obj.evaluate(templates)
            print("CB score of {} : {}".format(self.model_class, np.array(total_var).mean()))
            return total_mean
        elif(bias_type == 'gender'):
            log_gender_obj = LogProbabilityGender(self.model, self.tokenizer, self.device, 'masked', self.model_class, mask_token='[MASK]', dataset=dataset)
            return log_gender_obj.evaluate()
        elif(bias_type == 'race'):
            log_race_obj = LogProbabilityRace(self.model, self.tokenizer, self.device, self.model_class, 'masked', mask_token='[MASK]', dataset=dataset)
            return log_race_obj.evaluate()
        elif(bias_type == 'religion'):
            log_religion_obj = LogProbabilityReligion(self.model, self.tokenizer, self.device, self.model_class, 'masked', mask_token='[MASK]', dataset=dataset)
            return log_religion_obj.evaluate()
    def F1Score(self, bias_type='gender', dataset=None):
        if(bias_type == 'gender'):
            f1_obj = F1ScoreGender(self.model, self.tokenizer, self.device, self.model_class, 'masked', mask_token='[MASK]', dataset=dataset)
            return f1_obj.evaluate()
    def stereoSetScore(self, bias_type='gender', dataset=None, metric_type='full'):
        if(bias_type == 'gender'):
            stereoset_obj = StereoSetGender(self.model, self.tokenizer, self.device, self.model_class, 'masked', '[MASK]', dataset)
            return stereoset_obj.evaluate(metric_type)
        if(bias_type == 'race'):
            stereoset_obj = StereoSetRace(self.model, self.tokenizer, self.device, self.model_class, 'masked', '[MASK]', dataset)
            return stereoset_obj.evaluate(metric_type)
        if(bias_type == 'religion'):
            stereoset_obj = StereoSetReligion(self.model, self.tokenizer, self.device, self.model_class, 'masked', '[MASK]', dataset)
            return stereoset_obj.evaluate(metric_type)
        if(bias_type == 'profession'):
            stereoset_obj = StereoSetProfession(self.model, self.tokenizer, self.device, self.model_class, 'masked', '[MASK]', dataset)
            return stereoset_obj.evaluate(metric_type)
        if(bias_type == 'all'):
            stereoset_obj = StereoSetOverall(self.model, self.tokenizer, self.device, self.model_class, 'masked', '[MASK]', dataset)
            return stereoset_obj.evaluate(metric_type)

    def topKPercentage(self, bias_type='queer_nonqueer', dataset=None, k=5, lang='en', plot_graph=False):
        honest_obj = HonestMetric(self.model, self.tokenizer, self.device, self.model_class, 'masked', '[MASK]', dataset, lang, bias_type, k)
        honest_score, honest_df = honest_obj.evaluate(plot_graph)
        return honest_score, honest_df

    def WeatScore(self, bias_type='gender', dataset=None):
        if(bias_type == 'gender'):
            weat_obj = WeatScoreGender.WeatScoreGender(self.model, self.tokenizer, self.device, self.model_class, 'masked', '[MASK]', dataset)
            results = weat_obj.evaluate()
            p_score = 0
            e_score = 0
            counter = 0
            p_count = 0
            for result in results:
                if(result['p_value']<0.05):
                    p_score += result['p_value']
                    e_score += result['effect_size']
                    p_count+=1
            
                counter += 1
            
            #print(results)
            print('Percentage of p_value <0.05 ', p_count/counter)
            if(p_count>0):
                print('Average E-score ', e_score/p_count)
            else:
                print('Average E-score ', 0)
            return results
        if(bias_type == 'race'):
            weat_obj = WeatScoreRace.WeatScoreRace(self.model, self.tokenizer, self.device, self.model_class, 'masked', '[MASK]', dataset)
            results = weat_obj.evaluate()
            p_score = 0
            e_score = 0
            counter = 0
            p_count = 0
            for result in results:
                if(result['p_value']<0.05):
                    p_score += result['p_value']
                    e_score += result['effect_size']
                    p_count+=1
            
                counter += 1
            
            #print(results)
            print('Percentage of p_value <0.05 ', p_count/counter)
            if(p_count>0):
                print('Average E-score ', e_score/p_count)
            else:
                print('Average E-score ', 0)
            return results
        if(bias_type == 'religion'):
            weat_obj = WeatScoreReligion.WeatScoreReligion(self.model, self.tokenizer, self.device, self.model_class, 'masked', self.MSK, dataset)
            results = weat_obj.evaluate()
            p_score = 0
            e_score = 0
            counter = 0
            p_count = 0
            for result in results:
                if(result['p_value']<0.05):
                    p_score += result['p_value']
                    e_score += result['effect_size']
                    p_count+=1
            
                counter += 1
            
            #print(results)
            print('Percentage of p_value <0.05 ', p_count/counter)
            if(p_count>0):
                print('Average E-score ', e_score/p_count)
            else:
                print('Average E-score ', 0)
            return results
        if(bias_type == 'age'):
            weat_obj = WeatScoreAge.WeatScoreAge(self.model, self.tokenizer, self.device, self.model_class, 'masked', self.MSK, dataset)
            results = weat_obj.evaluate()
            p_score = 0
            e_score = 0
            counter = 0
            p_count = 0
            for result in results:
                if(result['p_value']<0.05):
                    p_score += result['p_value']
                    e_score += result['effect_size']
                    p_count+=1
            
                counter += 1
            
            #print(results)
            print('Percentage of p_value <0.05 ', p_count/counter)
            if(p_count>0):
                print('Average E-score ', e_score/p_count)
            else:
                print('Average E-score ', 0)
            return results
        if(bias_type == 'health'):
            weat_obj = WeatScoreHealth.WeatScoreHealth(self.model, self.tokenizer, self.device, self.model_class, 'masked', self.MSK, dataset)
            results = weat_obj.evaluate()
            p_score = 0
            e_score = 0
            counter = 0
            p_count = 0
            for result in results:
                if(result['p_value']<0.05):
                    p_score += result['p_value']
                    e_score += result['effect_size']
                    p_count+=1
            
                counter += 1
            
            #print(results)
            print('Percentage of p_value <0.05 ', p_count/counter)
            if(p_count>0):
                print('Average E-score ', e_score/p_count)
            else:
                print('Average E-score ', 0)
            return results
        return
    