import BiasMitigationMethods

maskedObj = BiasMitigationMethods.MaskedLMBiasMitigation(model_class='bert-base-uncased', use_pretrained=True)
maskedObj.genderFineTune()