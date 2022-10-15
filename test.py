from audioop import bias
import BiasDetection.BiasDetectionMetrics as BiasDetectionMetrics
import BiasMitigation.BiasMitigationMethods as BiasMitigationMethods
import sys

#Causal - gpt2, openai-gpt, ctrl, xlnet-base-cased, transfo-xl-wt103, xlm-mlm-en-2048, roberta-base
#Masked - bert-base-uncased, distilbert-base-uncased, roberta-base, albert-base-v1

#CausalMitObj = BiasMitigationMethods.CausalLMBiasMitigation(model_class='gpt2')
MaskedMitObj = BiasMitigationMethods.MaskedLMBiasMitigation(model_class='bert-base-uncased')
model, tokenizer = MaskedMitObj.SelfDebias('bert-base-uncased', 'BertForMaskedLM')
sys.path.insert(1, 'BiasDetection/')
maskedObj = BiasDetectionMetrics.MaskedLMBiasDetection(model_class='bert-base-uncased', use_pretrained=False, model = model._model, tokenizer =tokenizer)
#causalObj = BiasDetectionMetrics.CausalLMBiasDetection(model_class='gpt2', use_pretrained=False, model = model._model, tokenizer=tokenizer)
#causalObj.topKOverlap(bias_type='gender')
#causalObj.hellingerDistance()
#causalObj.WeatScore(bias_type='gender')
#causalObj.stereoSetScore(bias_type='all')
#causalObj.topKPercentage()
#causalObj.logProbability(bias_type='religion')

maskedObj.logProbability(bias_type='gender')
#maskedObj.F1Score(bias_type='gender')
#maskedObj.stereoSetScore(bias_type='all')
#maskedObj.topKPercentage()
#maskedObj.WeatScore(bias_type='health')



