import BiasDetectionMetrics

#maskedObj = BiasDetectionMetrics.MaskedLMBiasDetection(model_class='bert-base-uncased', use_pretrained=True)
causalObj = BiasDetectionMetrics.CausalLMBiasDetection(model_class='bert-base-uncased', use_pretrained=True)

#maskedObj.logProbability()
causalObj.hellingerDistanceSwapped()
#maskedObj.genderBiasProfessionF1Score()