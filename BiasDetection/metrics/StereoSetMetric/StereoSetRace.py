from BiasDetection.metrics.StereoSetMetric.StereoSet import StereoSet
from BiasDetection.metrics.StereoSetMetric.code.evaluation import parse_file
import sys
class StereoSetRace(StereoSet):
    def __init__(self, model, tokenizer, device, model_class, model_type, mask_token='[MASK]', dataset=None):
        super().__init__(model, tokenizer, device, model_class, model_type, mask_token, dataset)
    def intrasentence_bias(self):
        predictions = self.stereoObj.evaluate_intrasentence()
        parse_file(self.input_file,predictions, 'race')
        return
    def intersentence_bias(self):
        predictions = self.stereoObj.evaluate_intersentence()
        parse_file(self.input_file,predictions, 'race')
        return
    def stereoset_score(self):
        predictions_inter = self.stereoObj.evaluate_intersentence()
        predictions_intra = self.stereoObj.evaluate_intrasentence()
        parse_file(self.input_file,{'intrasentence':predictions_intra['intrasentence'], 'intersentence':predictions_inter['intersentence']}, 'race')
        return
    def evaluate(self, metric='full'):
        if(metric == 'intrasentence'):
            return self.intrasentence_bias()
        elif(metric == 'intersentence'):
            return self.intersentence_bias()
        else:
            return self.stereoset_score()