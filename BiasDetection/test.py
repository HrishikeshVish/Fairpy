import BiasDetectionMetrics

maskedObj = BiasDetectionMetrics.MaskedLMBiasDetection(model_class='bert-base-uncased', use_pretrained=True)
causalObj = BiasDetectionMetrics.CausalLMBiasDetection(model_class='gpt2', use_pretrained=True)

#maskedObj.logProbability()
#causalObj.topKOverlap()
maskedObj.genderBiasProfessionF1Score()