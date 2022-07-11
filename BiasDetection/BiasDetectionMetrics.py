from abc import ABC, abstractmethod
from BiasMaskedLM.masked_metrics import log_probability_for_multiple_sentence
from BiasCausalLM.LocalBias.measure_local_bias import topk_overlap, hellinger_distance_between_bias_swapped_context, probabiliy_of_real_next_token
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer
from transformers import (
    CTRLLMHeadModel, CTRLTokenizer,
    GPT2LMHeadModel, GPT2Tokenizer,
    OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
    TransfoXLLMHeadModel, TransfoXLTokenizer,
    XLMTokenizer, XLMWithLMHeadModel,
    XLNetLMHeadModel, XLNetTokenizer,
    BertForMaskedLM, BertTokenizer,
    DistilBertForMaskedLM, DistilBertTokenizer,
    RobertaForMaskedLM, RobertaTokenizer,
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
        }
        self.config = ''
        self.model, self.tokenizer = self.load_model(model_class, model_path, use_pretrained)
        self.embedding = self.model.lm_head.weight.cpu().detach().numpy()
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
        topk_overlap(self.model, self.tokenizer, self.embedding, self.device)
        return
    def hellingerDistanceSwapped(self):
        hellinger_distance_between_bias_swapped_context(self.model, self.tokenizer, self.embedding, self.device)
        return
    def probPrediction(self):
        probabiliy_of_real_next_token(self.model, self.tokenizer, self.embedding, self.device)
        return

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
            }
        self.model, self.tokenizer = self.load_model(model_class, model_path, use_pretrained)
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
    
    def logProbability(self, templates=None):
        if(templates == None):
            total_mean, total_var, total_std = log_probability_for_multiple_sentence(self.model, self.tokenizer, self.device, self.MSK, use_pretrained=self.use_pretrained)
        else:
            total_mean, total_var, total_std = log_probability_for_multiple_sentence(self.model, self.tokenizer, self.device, self.MSK, templates, use_pretrained=self.use_pretrained)
        print("CB score of {} : {}".format(self.model_class, np.array(total_var).mean()))
        return total_mean, total_var, total_std
