from honest import honest
import pandas as pd
from transformers import pipeline, set_seed
import plotly.graph_objects as go
class HonestMetric:
    def __init__(self, model, tokenizer, device, model_class, model_type = 'causal', mask_token='[MASK]',  dataset=None, language='en', bias_type='queer_nonqueer', k=5):
        self.model_name=model_class
        self.model_type = model_type
        self.model = model
        self.tokenizer = tokenizer
        self.language = language
        self.bias_type = bias_type
        self.device = device
        self.mask_token = mask_token
        self.dataset = dataset
        self.k = k
        self.evaluator = honest.HonestEvaluator(self.language)
        self.masked_templates = self.evaluator.templates(data_set=self.bias_type)

        return
    def evaluateCausal(self, plot_results=False):
        a = pd.DataFrame(self.masked_templates).T
        a = a[a['number'] =='singular']
        generator = pipeline('text-generation', model = self.model_name, device=0)
        set_seed(42)
        filled_templates = [[fill['generated_text'][len(masked_sentence.replace(' [M]','')):] for fill in generator(masked_sentence.replace(' [M].',''), max_length=len(self.tokenizer(masked_sentence.replace(' [M].',''))['input_ids'])+10, num_return_sequences=self.k)] for masked_sentence in self.masked_templates.keys()]
        honest_score, honest_df = self.evaluator.honest_dataframe(filled_templates, self.masked_templates)
        print(self.model_name, self.k, honest_score)
        if(plot_results):
            hurtlex_cat = ['animals', 'crime and immoral behavior', 'derogatory words', 'female genitalia', 'male genitalia', 'homosexuality', 'potential negative connotations', 'professions and occupations', 'prostitution', 'social and economic disadvantage']
            hurtlex_cat_names = ['an', 're', 'cds', 'asf', 'asm', 'om', 'qas', 'pa', 'pr', 'is']
            df_identity = honest_df.groupby('category')[hurtlex_cat_names].sum()
            df_count = honest_df.groupby('category')[hurtlex_cat_names].count()*self.k
            df_perc = df_identity/df_count*100

            plots = [go.Scatterpolar(r=df_perc.loc[i], theta=hurtlex_cat, fill='toself',
                         name=i) for i, row in df_perc.iterrows()]

            fig = go.Figure(
            data=plots,
            layout=go.Layout(
                #title=go.layout.Title('hey'),
                polar={'radialaxis': {'visible': True}},
                #showlegend=True
            )
            )
        return honest_score, honest_df
    def get_completion(self, sentence, template):
        start = sentence.find(template[:-4])+len(template[:-4])
        if self.tokenizer.eos_token != None and self.tokenizer.eos_token in sentence:
            end = sentence.find(self.tokenizer.eos_token)
        elif self.tokenizer.sep_token != None and self.tokenizer.sep_token in sentence:
            end = sentence.find(self.tokenizer.sep_token)
        else:
            end = len(sentence)-1

        return sentence[start:end]
    def evaluateMasked(self, plot_results=False):
        nlp_fill = pipeline('fill-mask', model=self.model, tokenizer=self.tokenizer, top_k = self.k, device=0)
        filled_templates = [[self.get_completion(fill['sequence'],masked_sentence) for fill in nlp_fill(masked_sentence.replace('[M]',self.tokenizer.mask_token))] for masked_sentence in self.masked_templates.keys()]
        
        honest_score, honest_df = self.evaluator.honest_dataframe(filled_templates, self.masked_templates)
        print(self.model_name, self.k, honest_score)
        if(plot_results==True):
            hurtlex_cat = ['animals', 'crime and immoral behavior', 'derogatory words', 'female genitalia', 'male genitalia', 'homosexuality', 'potential negative connotations', 'professions and occupations', 'prostitution', 'social and economic disadvantage']
            hurtlex_cat_names = ['an', 're', 'cds', 'asf', 'asm', 'om', 'qas', 'pa', 'pr', 'is']
            df_identity = honest_df.groupby('category')[hurtlex_cat_names].sum()
            df_count = honest_df.groupby('category')[hurtlex_cat_names].count()*self.k
            df_perc = df_identity/df_count*100
            plots = [go.Scatterpolar(r=df_perc.loc[i], theta=hurtlex_cat, fill='toself',
                                     name=i) for i, row in df_perc.iterrows()]
            fig = go.Figure(
                data=plots,
                layout=go.Layout(
                polar={'radialaxis': {'visible': True}}
                )
            )
        return honest_score, honest_df
    
    def evaluate(self, plot_results):
        if(self.model_type == 'causal'):
            return self.evaluateCausal(plot_results)
        else:
            return self.evaluateMasked(plot_results)



