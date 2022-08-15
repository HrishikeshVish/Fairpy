import BiasMitigationMethods
from techniques.GenderAugmentRetrain import Augment_utils
import sys
sys.path.insert(3, 'techniques/UpstreamMitigation')
maskedObj = BiasMitigationMethods.MaskedLMBiasMitigation(model_class='bert-base-uncased', use_pretrained=True)
#maskedObj.FineTune(dataset = 'yelp_med')
#maskedObj.MiscWordAugment(['king', 'queen'],['monarch', 'monarch'])
#causalObj = BiasMitigationMethods.CausalLMBiasMitigation(model_class='xlnet-base-cased', use_pretrained=True)
#causalObj.SelfDebias('bert-base-uncased', 'BertForMaskedLM')

#Augment_utils.ethnicity_counterfactual_augmentation([''])
#causalObj.DiffPruning()
#maskedObj.EntropyAttentionRegularization()
maskedObj.UpstreamBiasMitigation()