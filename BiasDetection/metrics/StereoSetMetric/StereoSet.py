from BiasDetection.metrics.StereoSetMetric.code.eval_generative_models import BiasEvaluator as causalBiasEvaluator
from BiasDetection.metrics.StereoSetMetric.code.eval_discriminative_models import BiasEvaluator as maskedBiasEvaluator
import sys
class StereoSet:
    def __init__(self, model, tokenizer, device, model_class,model_type, mask_token='[MASK]', dataset=None):
        self.input_file = sys.path[1]+'data/StereoSetData/dev.json'
        if(model_type == 'causal'):
            self.stereoObj = causalBiasEvaluator(model, device, pretrained_class=model_class, tokenizer=tokenizer, input_file=self.input_file)
        else:
            self.stereoObj = maskedBiasEvaluator(model, device, pretrained_class=model_class, tokenizer=tokenizer, input_file=self.input_file)
    