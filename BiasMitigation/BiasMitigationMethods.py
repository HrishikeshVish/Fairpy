from abc import ABC, abstractmethod
from audioop import bias
import sys
from genderAugmentRetrain.masked_finetune_gender import fineTune as gender_tune
from LMRetrain.causalLMRetrain import Retrain as causalRetrain
from LMRetrain.maskedLMRetrain import Retrain as maskedRetrain
from NullSpaceProjection.inlp_projection_matrix import ComputeProjectionMatrix
from SentenceDebias.sentence_debias_subspace import sentence_debias
import models
import json
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
        self.retrain_sets = {'wikipedia2.5':"data/text/wikipedia-2.5.txt", 'wikipedia10':"data/text/wikipedia-10.txt", 
                             'news100': "data/text_corpus/news_100.txt", "news200":"data/text_corpus/news_200.txt", "reddit":"data/text_corpus/reddit.txt",
                             "wikitext":"data/text_corpus/wikitext.txt", "yelp_sm":"data/text_corpus/yelp_review_1mb.txt",
                             "yelp_med":"data/text_corpus/yelp_review_5mb.txt", "yelp_lg":"data/text_corpus/yelp_review_10mb.txt"}
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
    
    def DropOutDebias(self, model_class, bias_type='gender', train_data='yelp_sm', epochs=100):
        if(train_data not in self.retrain_sets.keys()):
            train_data = self.retrain_sets['yelp_sm']
        else:
            train_data = self.retrain_sets[train_data]
        causalRetrain(model_name_or_path=model_class, output_dir='savedModel/', train_file=train_data, counterfactual_augmentation=bias_type, do_train=True, seed=4, 
                preprocessing_num_workers=4, max_seq_length=512, save_steps=500, max_steps=epochs, per_device_train_batch_size=32, gradient_accumulation_steps=16,
                dropout_debias=True)
        return
    def NullSpaceProjection(self, model_class, huggingface_class, bias_type, train_data='yelp_sm'):
        #model, tokenizer = self.load_model(model_class, self.model_path, True)
        model = getattr(models, huggingface_class)(model_class)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_class)
        dataset = self.retrain_sets['yelp_sm']
        if(train_data in self.retrain_sets.keys()):
            dataset = self.retrain_sets[train_data]
        projection_matrix = ComputeProjectionMatrix(model, tokenizer, model_class, dataset, train_data, bias_type)
        model = getattr(models, "INLP"+huggingface_class)(model_class, projection_matrix)
        return model, tokenizer
    def SentenceDebias(self, model_class, huggingface_class, bias_type, train_data='yelp_sm'):
        model = getattr(models, huggingface_class)(model_class)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_class)
        dataset = self.retrain_sets['yelp_sm']
        if(train_data in self.retrain_sets.keys()):
            dataset = self.retrain_sets[train_data]
        bias_direction = sentence_debias(model, tokenizer, model_class, dataset, train_data, bias_type)
        model = getattr(models, "SentenceDebias"+huggingface_class)(model_class, bias_direction)
        return model, tokenizer
    def SelfDebias(self, model_class, huggingface_class):
        model = getattr(models, 'SelfDebias'+huggingface_class)(model_class)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_class)
        return model, tokenizer
    def AddSocialConstructs(self, subject='they', object='them', poss_obj='their', poss_pro='theirs', reflexive='themself'):
        attribute_file = f"data/bias_attribute_words.json"
        with open(attribute_file, "rb") as f:
            bias_attribute_words = json.load(f)
        print(bias_attribute_words)
        bias_attribute_words['non-binary'][0].append(object)
        bias_attribute_words['non-binary'][1].append(object)
        bias_attribute_words['non-binary'][2].append(subject)
        bias_attribute_words['non-binary'][3].append(subject)
        bias_attribute_words['non-binary'][4].append(reflexive)
        bias_attribute_words['non-binary'][5].append(reflexive)
        bias_attribute_words['non-binary'][6].append(poss_pro)
        bias_attribute_words['non-binary'][7].append(poss_pro)
        f = open(attribute_file, 'w', encoding='utf-8')
        json.dump(bias_attribute_words, f, indent=3)
    def MiscWordAugment(self, word_list, augment_list, construct_name='misc_social_construct'):
        if(len(word_list)!=len(augment_list)):
            print("List sizes not equal, will exit")
            return
        else:
            attribute_file = f"data/bias_attribute_words.json"
            with open(attribute_file, "rb") as f:
                bias_attribute_words = json.load(f)
            #print(bias_attribute_words)
            augment_data = [[word_list[i], augment_list[i]] for i in range(len(word_list))]
            bias_attribute_words[construct_name] = augment_data
            f = open(attribute_file, 'w', encoding='utf-8')
            json.dump(bias_attribute_words, f, indent=3)



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
        self.retrain_sets = {'wikipedia2.5':"data/text/wikipedia-2.5.txt", 'wikipedia10':"data/text/wikipedia-10.txt", 
                             'news100': "data/text_corpus/news_100.txt", "news200":"data/text_corpus/news_200.txt", "reddit":"data/text_corpus/reddit.txt",
                             "wikitext":"data/text_corpus/wikitext.txt", "yelp_sm":"data/text_corpus/yelp_review_1mb.txt",
                             "yelp_med":"data/text_corpus/yelp_review_5mb.txt", "yelp_lg":"data/text_corpus/yelp_review_10mb.txt"}
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
    def FineTune(self, dataset='yelp_sm', bias_type='gender'):
        if(dataset in self.retrain_sets.keys()):
            dataset_location = self.retrain_sets[dataset]
        else:
            dataset_location= self.retrain_sets['yelp_sm']
        model = gender_tune(self.device, self.model, self.tokenizer, dataset, dataset_location, bias_type)
        return model
    def DropOutDebias(self, model_class, bias_type='gender', train_data='yelp_sm', epochs=100):
        if(train_data not in self.retrain_sets.keys()):
            train_data = self.retrain_sets['yelp_sm']
        else:
            train_data = self.retrain_sets[train_data]
        maskedRetrain(model_name_or_path=model_class, output_dir='savedModel/', train_file=train_data, counterfactual_augmentation=bias_type, do_train=True, seed=4, 
                preprocessing_num_workers=4, max_seq_length=512, save_steps=500, max_steps=epochs, per_device_train_batch_size=32, gradient_accumulation_steps=16,
                dropout_debias=True)
        return
    def NullSpaceProjection(self, model_class, huggingface_class, bias_type, train_data='yelp_sm'):
        #model, tokenizer = self.load_model(model_class, self.model_path, True)
        model = getattr(models, huggingface_class)(model_class)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_class)
        dataset = self.retrain_sets['yelp_sm']
        if(train_data in self.retrain_sets.keys()):
            dataset = self.retrain_sets[train_data]
        projection_matrix = ComputeProjectionMatrix(model, tokenizer, model_class, dataset, train_data, bias_type)
        model = getattr(models, "INLP"+huggingface_class)(model_class, projection_matrix)
        return model, tokenizer
    def SentenceDebias(self, model_class, huggingface_class, bias_type, train_data='yelp_sm'):
        model = getattr(models, huggingface_class)(model_class)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_class)
        dataset = self.retrain_sets['yelp_sm']
        if(train_data in self.retrain_sets.keys()):
            dataset = self.retrain_sets[train_data]
        bias_direction = sentence_debias(model, tokenizer, model_class, dataset, train_data, bias_type)
        model = getattr(models, "SentenceDebias"+huggingface_class)(model_class, bias_direction)
        return model, tokenizer
    def SelfDebias(self, model_class, huggingface_class):
        model = getattr(models, 'SelfDebias'+huggingface_class)(model_class)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_class)
        return model, tokenizer
    def AddSocialConstructs(self, subject='they', object='them', poss_obj='their', poss_pro='theirs', reflexive='themself'):
        attribute_file = f"data/bias_attribute_words.json"
        with open(attribute_file, "rb") as f:
            bias_attribute_words = json.load(f)
        print(bias_attribute_words)
        bias_attribute_words['non-binary'][0].append(object)
        bias_attribute_words['non-binary'][1].append(object)
        bias_attribute_words['non-binary'][2].append(subject)
        bias_attribute_words['non-binary'][3].append(subject)
        bias_attribute_words['non-binary'][4].append(reflexive)
        bias_attribute_words['non-binary'][5].append(reflexive)
        bias_attribute_words['non-binary'][6].append(poss_pro)
        bias_attribute_words['non-binary'][7].append(poss_pro)
        f = open(attribute_file, 'w', encoding='utf-8')
        json.dump(bias_attribute_words, f, indent=3)
    def MiscWordAugment(self, word_list, augment_list, construct_name='misc_social_construct'):
        if(len(word_list)!=len(augment_list)):
            print("List sizes not equal, will exit")
            return
        else:
            attribute_file = f"data/bias_attribute_words.json"
            with open(attribute_file, "rb") as f:
                bias_attribute_words = json.load(f)
            #print(bias_attribute_words)
            augment_data = [[word_list[i], augment_list[i]] for i in range(len(word_list))]
            bias_attribute_words[construct_name] = augment_data
            f = open(attribute_file, 'w', encoding='utf-8')
            json.dump(bias_attribute_words, f, indent=3)

