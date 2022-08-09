from BiasDetection.metrics.WeatScore.WeatScore import WeatScore
class WeatScoreAge(WeatScore):
    def __init__(self, model, tokenizer, device, model_class, model_type, mask_token='[MASK]',dataset=None):
        super().__init__(model, tokenizer, device, model_class, model_type, mask_token='[MASK]', dataset=None)
        self.path = self.path + '/age/'
    def evaluate(self):
        return super().SeatScore(self.path)

    