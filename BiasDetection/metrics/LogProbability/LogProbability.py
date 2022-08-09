class LogProbability:
    def __init__(self, model, tokenizer, device, model_class, model_type, mask_token='[MASK]', dataset=None):
        self.crows_path = 'BiasDetection/data/crows/crows_pairs_anonymized.csv'
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_class = model_class
        self.model_type = model_type
        self.mask_token = mask_token
        self.dataset = dataset
    