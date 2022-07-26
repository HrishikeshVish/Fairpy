import BiasDetectionMetrics

#maskedObj = BiasDetectionMetrics.MaskedLMBiasDetection(model_class='bert-base-uncased', use_pretrained=True)
causalObj = BiasDetectionMetrics.MaskedLMBiasDetection(model_class='bert-base-uncased', use_pretrained=True)
causalObj.intrasentenceBias()

#maskedObj.logProbability()
#causalObj.hellingerDistanceSwapped()
#maskedObj.genderBiasProfessionF1Score()

