import argparse
import BiasDetectionMetrics
parser = argparse.ArgumentParser(description='Toolkit for detection and Mitigation of biases in Large Pretrained Language Models')
parser.add_argument('--model_class', help='Type of Model')
parser.add_argument('--metric', help='Bias Detection Metric')
parser.add_argument('--data', help='Bias Dataset')
args = parser.parse_args()

if(args.metric == 'k-overlap'):
    obj = BiasDetectionMetrics.CausalLMBiasDetection(model_class=args.model_class, use_pretrained=True)
    obj.topKOverlap()
elif(args.metric == 'hellinger-dist'):
    obj = BiasDetectionMetrics.CausalLMBiasDetection(model_class=args.model_class, use_pretrained=True)
    obj.hellingerDistanceSwapped()
elif(args.metric == 'prediction-prob'):
    obj = BiasDetectionMetrics.CausalLMBiasDetection(model_class=args.model_class, use_pretrained=True)
    obj.probPrediction()
elif(args.metric == 'log-probability'):
    obj = BiasDetectionMetrics.MaskedLMBiasDetection(model_class=args.model_class, use_pretrained=True)
    obj.logProbability()
elif(args.metric == 'f1-score'):
    obj = BiasDetectionMetrics.MaskedLMBiasDetection(model_class=args.model_class, use_pretrained=True)
    obj.genderBiasProfessionF1Score()
