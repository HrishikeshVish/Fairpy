from audioop import bias
import BiasDetection.BiasDetectionMetrics as BiasDetectionMetrics

maskedObj = BiasDetectionMetrics.MaskedLMBiasDetection(model_class='bert-base-uncased', use_pretrained=True)
causalObj = BiasDetectionMetrics.CausalLMBiasDetection(model_class='gpt2', use_pretrained=True)
maskedObj.StereoSetScore()


#maskedObj.logProbability(bias_type='gender')
#maskedObj.logProbability(bias_type='nationality')
#causalObj.hellingerDistanceSwapped()
#maskedObj.genderBiasProfessionF1Score()

