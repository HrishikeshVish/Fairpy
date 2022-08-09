from BiasDetection.metrics.LogProbability.LogProbability import LogProbability
from BiasDetection.metrics.LogProbability.crows.crows import CrowSPairsRunner
class LogProbabilityRace(LogProbability):
    def __init__(self, model, tokenizer, device, model_class, model_type, mask_token='[MASK]', dataset=None):
        super().__init__(model, tokenizer, device, model_class, model_type, mask_token, dataset)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_class = model_class
        self.mask_token = mask_token
        self.dataset = dataset
        return
    def LogProbabilityCrows(self, model, tokenizer, device, model_type):
        is_generative = True
        if(model_type == 'masked'):
            is_generative = False
        runner = CrowSPairsRunner(
            model=model,
            tokenizer=tokenizer,
            input_file=self.crows_path,
            bias_type='race',
            is_generative=is_generative  # Affects model scoring.
        )
        results = runner()

        print(f"Metric: {results}")
        return results
    def evaluate(self):
        return self.LogProbabilityCrows(self.model, self.tokenizer, self.device, self.model_type)