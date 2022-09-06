from audioop import bias
import BiasDetection.BiasDetectionMetrics as BiasDetectionMetrics

#Causal - gpt2, openai-gpt, ctrl, xlnet-base-cased, transfo-xl-wt103, xlm-mlm-en-2048, roberta-base
#Masked - bert-base-uncased, distilbert-base-uncased, roberta-base, albert-base-v1

maskedObj = BiasDetectionMetrics.MaskedLMBiasDetection(model_class='bert-base-uncased', use_pretrained=True)
#causalObj = BiasDetectionMetrics.CausalLMBiasDetection(model_class='transfo-xl-wt103', use_pretrained=True)
#causalObj.topKOverlap(bias_type='gender')
#causalObj.hellingerDistance()
#causalObj.WeatScore(bias_type='gender')
#causalObj.stereoSetScore(bias_type='all')
#causalObj.topKPercentage()
#causalObj.logProbability(bias_type='religion')

#maskedObj.logProbability(bias_type='gender')
#maskedObj.F1Score(bias_type='gender')
#maskedObj.stereoSetScore(bias_type='all')
#maskedObj.topKPercentage()
maskedObj.WeatScore(bias_type='health')



