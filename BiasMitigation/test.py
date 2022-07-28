import BiasMitigationMethods

maskedObj = BiasMitigationMethods.MaskedLMBiasMitigation(model_class='bert-base-uncased', use_pretrained=True)
maskedObj.FineTune(dataset = 'yelp_med')

#causalObj = BiasMitigationMethods.CausalLMBiasMitigation(model_class='gpt2', use_pretrained=True)
#causalObj.SelfDebias('bert-base-uncased', 'BertForMaskedLM')