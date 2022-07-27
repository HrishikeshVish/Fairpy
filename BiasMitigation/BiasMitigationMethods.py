from abc import ABC, abstractmethod
import sys
from genderAugmentRetrain.masked_finetune_gender import fineTune as gender_tune
sys.path.insert(2, '')

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
class LMBiasMitigation(ABC):
    def __init__(self, model_class, model_path, write_to_file, use_pretrained):
        self.use_pretrained = use_pretrained
        self.write_to_file = write_to_file
        self.model_class = model_class
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    @abstractmethod
    def load_model(self):
        pass

class CausalLMBiasMitigation(LMBiasMitigation):

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

class MaskedLMBiasMitigation(LMBiasMitigation):
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
    def genderFineTune(self, dataset):
        model = gender_tune(self.device, self.model, self.tokenizer, dataset)

