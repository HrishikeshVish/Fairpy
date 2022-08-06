from BiasDetection.metrics.StereoSetMetric.StereoSet import StereoSet
from BiasDetection.metrics.StereoSetMetric.code.evaluation import parse_file
import sys
class StereoSetOverall(StereoSet):
    def __init__(self, model, device, pretrained_class, tokenizer, input_file, model_type):
        super().__init__(model, device, pretrained_class, tokenizer, input_file, model_type)
        self.input_file = input_file
    def intrasentence_bias(self):
        predictions = self.stereoObj.evaluate_intrasentence()
        parse_file(self.input_file,predictions, 'overall')
        return
    def intersentence_bias(self):
        predictions = self.stereoObj.evaluate_intersentence()
        parse_file(self.input_file,predictions, 'overall')
        return
    def stereoset_score(self):
        predictions_inter = self.stereoObj.evaluate_intersentence()
        predictions_intra = self.stereoObj.evaluate_intrasentence()
        parse_file(self.input_file,{'intrasentence':predictions_intra['intrasentence'], 'intersentence':predictions_inter['intersentence']}, 'overall')
        return