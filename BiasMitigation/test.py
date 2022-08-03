import BiasMitigationMethods
from genderAugmentRetrain import Augment_utils
maskedObj = BiasMitigationMethods.MaskedLMBiasMitigation(model_class='bert-base-uncased', use_pretrained=True)
#maskedObj.FineTune(dataset = 'yelp_med')
maskedObj.MiscWordAugment(['king', 'queen'],['monarch', 'monarch'])
#causalObj = BiasMitigationMethods.CausalLMBiasMitigation(model_class='gpt2', use_pretrained=True)
#causalObj.SelfDebias('bert-base-uncased', 'BertForMaskedLM')

#Augment_utils.ethnicity_counterfactual_augmentation([''])