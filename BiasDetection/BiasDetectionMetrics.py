from abc import ABC, abstractmethod
import sys
sys.path.insert(1, 'BiasDetection/')
from BiasMaskedLM.masked_metrics_nationality import log_probability_for_multiple_sentence
from BiasMaskedLM.masked_metrics_gender import log_probability_gender, f1_score_gender_profession
from BiasCausalLM.LocalBias.measure_local_bias import topk_overlap, hellinger_distance_between_bias_swapped_context, probabiliy_of_real_next_token
from StereoSet.code.eval_generative_models import BiasEvaluator as generativeBiasEval
from StereoSet.code.eval_discriminative_models import BiasEvaluator as discriminativeBiasEval
from StereoSet.code.eval_sentiment_models import BiasEvaluator as sentimentBiasEval
from StereoSet.code.evaluation import parse_file
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

    def __init__(self, model_class='',model_path='', write_to_file=False, use_pretrained=True):
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
        self.config = ''
        self.model, self.tokenizer = self.load_model(model_class, model_path, use_pretrained)
        self.stereoSet = generativeBiasEval(self.model, self.device, tokenizer = self.tokenizer, input_file=sys.path[1]+'StereoSet/data/dev.json')
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
    
    def topKOverlap(self):
        topk_overlap(self.model, self.tokenizer, self.embedding, self.device, self.transformer)
        return
    def hellingerDistanceSwapped(self):
        hellinger_distance_between_bias_swapped_context(self.model, self.tokenizer, self.embedding, self.device, self.transformer)
        return
    def probPrediction(self):
        probabiliy_of_real_next_token(self.model, self.tokenizer, self.embedding, self.device, self.transformer)
        return
    def intersentenceBias(self):
        predictions = self.stereoSet.evaluate_intersentence()
        parse_file(sys.path[1]+'StereoSet/data/dev.json',predictions)
        return
    def intrasentenceBias(self):
        predictions = self.stereoSet.evaluate_intrasentence()
        parse_file(sys.path[1]+'StereoSet/data/dev.json',predictions)
        return
    def StereoSetScore(self):
        predictions_inter = self.stereoSet.evaluate_intersentence()
        predictions_intra = self.stereoSet.evaluate_intrasentence()
        parse_file(sys.path[1]+'StereoSet/data/dev.json',{'intrasentence':predictions_intra['intrasentence'], 'intersentence':predictions_inter['intersentence']})
        

class MaskedLMBiasDetection(LMBiasDetection):
    def __init__(self, model_class='',model_path='', write_to_file=False, use_pretrained=True):
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
        self.model, self.tokenizer = self.load_model(model_class, model_path, use_pretrained)
        self.stereoSet = discriminativeBiasEval(self.model, self.device, tokenizer = self.tokenizer, input_file=sys.path[1]+'StereoSet/data/dev.json')
        self.config = ''
        self.MSK = '[MASK]'
        if('roberta' in model_class):
            self.MSK = '<mask>'
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
    
    def logProbability(self, bias_type='gender', templates=None):
        if(bias_type == 'nationality'):
            if(templates == None):
                total_mean, total_var, total_std = log_probability_for_multiple_sentence(self.model, self.tokenizer, self.device, self.MSK, use_pretrained=self.use_pretrained)
            else:
                total_mean, total_var, total_std = log_probability_for_multiple_sentence(self.model, self.tokenizer, self.device, self.MSK, templates, use_pretrained=self.use_pretrained)
            print("CB score of {} : {}".format(self.model_class, np.array(total_var).mean()))
            return total_mean
        elif(bias_type == 'gender'):
            associations = log_probability_gender(self.model, self.tokenizer, self.device)
            print("Mean Probability Score : {}".format(np.array(associations).mean()))

    def genderBiasProfessionF1Score(self):
        results = f1_score_gender_profession(self.model, self.tokenizer, self.device, self.MSK, self.model_class)
        return results
    def intersentenceBias(self):
        predictions = self.stereoSet.evaluate_intersentence()
        parse_file(sys.path[1]+'StereoSet/data/dev.json',predictions)
        return
    def intrasentenceBias(self):
        predictions = self.stereoSet.evaluate_intrasentence()
        parse_file(sys.path[1]+'StereoSet/data/dev.json',predictions)
        return
    def StereoSetScore(self):
        predictions_inter = self.stereoSet.evaluate_intersentence()
        predictions_intra = self.stereoSet.evaluate_intrasentence()
        parse_file(sys.path[1]+'StereoSet/data/dev.json',{'intrasentence':predictions_intra['intrasentence'], 'intersentence':predictions_inter['intersentence']})