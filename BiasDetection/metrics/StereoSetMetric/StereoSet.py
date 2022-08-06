from BiasDetection.metrics.StereoSetMetric.code.eval_generative_models import BiasEvaluator as causalBiasEvaluator
from BiasDetection.metrics.StereoSetMetric.code.eval_discriminative_models import BiasEvaluator as maskedBiasEvaluator
class StereoSet:
    def __init__(self, model, device, pretrained_class, tokenizer, input_file, model_type):
        if(model_type == 'causal'):
            self.stereoObj = causalBiasEvaluator(model, device, pretrained_class=pretrained_class, tokenizer=tokenizer, input_file=input_file)
        else:
            self.stereoObj = maskedBiasEvaluator(model, device, pretrained_class=pretrained_class, tokenizer=tokenizer, input_file=input_file)
    