from BiasDetection.metrics.WeatScore.code.seat import SEATRunner
from transformers import AutoTokenizer
class WeatScore:
    def __init__(self, model, tokenizer, device, model_class, model_type, mask_token='[MASK]', dataset=None):
        self.model = model
        self.tokenizer = tokenizer
        self.path = 'BiasDetection/data/weatStereotypes'
        self.device = device
        self.model_class = model_class
        self.mask_token = mask_token
        self.dataset = dataset
        self.model_type = model_type
    def SeatScore(self, path):
        seatObj = SEATRunner(self.model, self.tokenizer, self.device, path)
        return seatObj()
        

