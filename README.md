<div align="center">
  <img src="/images/Fairpy_Logo.png">
</div>

![Python](https://img.shields.io/badge/Python-3.6,%203.7,%203.8,%203.9.%203.10-red?style=for-the-badge&logo=python)
![Huggingface](https://img.shields.io/badge/Transformers-4.21.0-blue?style=for-the-badge&logo=openai)
![Pytorch](https://img.shields.io/badge/Pytorch-1.12.0-yellow?style=for-the-badge&logo=pytorch)
![Cuda](https://img.shields.io/badge/Cuda-11.6-green?style=for-the-badge&logo=nvidia)
![SkLearn](https://img.shields.io/badge/Sklearn-1.0.2-green?style=for-the-badge&logo=scikit-learn)
![Scipy](https://img.shields.io/badge/Scipy-1.7.3-brown?style=for-the-badge&logo=scipy)
![Keras](https://img.shields.io/badge/Keras-2.7.0-blue?style=for-the-badge&logo=keras)
![Tensorflow](https://img.shields.io/badge/Tensorflow-2.7.0-orange?style=for-the-badge&logo=tensorflow)
![NLTK](https://img.shields.io/badge/NLTK-3.6.7-pink?style=for-the-badge&logo=python)

Fairpy is an open source Toolkit for measuring and mitigating biases in large pretrained language models. It currently supports a wide range of bias detection tools and Bias Mitigation techniques along with interfaces for augmenting corpus, plugging in custom language models and extending the package to include new techniques. 

Paper Link https://arxiv.org/abs/2302.05508

## Features

- Bias Detection Methods
  - Top K Overlap
  - Hellinger Distance
  - F1 Score
  - Honest
  - Log Probability
  - StereoSet
  - WEAT/SEAT Score
- Bias Mitigation Methods
  - DiffPruning
  - Entropy Attention Regularization
  - Counter Factual Data Augmentation
  - Null Space Projection
  - Self Debias (Incomplete)
  - Dropout Regularization
- Models
  - CTRL
  - GPT-2
  - OpenAI-GPT
  - TransfoXL
  - BERT
  - DistilBERT
  - RoBERTa
  - AlBERT
  - XLM
  - XLNet

## Usage
### Bias Detection in Masked Language Models
```python
from fairpy import BiasDetection.BiasDetectionMetrics as BiasDetectionMetrics
maskedObj = BiasDetectionMetrics.MaskedLMBiasDetection(model_class = 'bert-base-uncased')
maskedObj.WeatScore(bias_type='health')
```
### Bias Detection in Causal Language Models
```python
from fairpy import BiasDetection.BiasDetectionMetrics as BiasDetectionMetrics
causalObj = BiasDetectionMetrics.CausalLMBiasDetection(model_class = 'gpt2')
causalObj.stereoSetScore(bias_type='gender')
```
### Bias Mitigation in Masked Language Models
```python
from fairpy import BiasMitigation.BiasMitigationMethods as BiasMitigationMethods
MaskedMitObj = BiasMitigationMethods.MaskedLMBiasMitigation(model_class='bert-base-uncased')
model, tokenizer = MaskedMitObj.NullSpaceProjection('bert-base-uncased', 'BertForMaskedLM', 'race', train_data='yelp_sm')
```
### Bias Detection in Causal Language Models
```python
from fairpy import BiasMitigation.BiasMitigationMethods as BiasMitigationMethods
CausalMitObj = BiasMitigationMethods.CausalLMBiasMitigation(model_class='gpt2')
model, tokenizer = CausalMitObj.DropOutDebias('gpt2', 'religion', train_data='yelp_sm')
```
