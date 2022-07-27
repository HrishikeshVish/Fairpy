import csv
from glob import glob
import os
from tqdm import tqdm

import nltk
nltk.download('names')
from nltk.corpus import names

import pandas as pd

def load_data(path):
    """
    Load content from csv's as a list of lists, with each sublist
    correspoinding to a line in the csv.
    """
    content = []
    with open(path) as f:
        reader = csv.reader(f, delimiter=",")
        for line in reader:
            if len(line) > 0: 
                if line[0][0] != '#':
                    content.append(line)
            else:
                content.append([])
    return content



def generate_pronoun_map():
    """
    Create pronoun mapping to switch possessive
    and personal pronouns to their opposite gender
    """
    pronoun_map_df = pd.DataFrame([
        ['he', '[she]', 'PRP'],
        ['she', '[he]', 'PRP'],
        ['his', '[her]', 'PRP$'],   
        ['his', '[hers]', 'PRP'],
        ['hers', '[his]', 'PRP'], # Added to counter line 5026 in 'bc/phoenix/00/phoenix_0000.gold_conll.csv'
        ['her', '[his]', 'PRP$'],   
        ['him', '[her]', 'PRP'],
        ['her', '[him]', 'PRP'],
    ])
    pronoun_map_df.columns = ['word', 'flipped_pronoun', 'pos_0']
    return pronoun_map_df

def preprocess_content(data):
    """
    Select "word" and Part of Speech column ("pos_0") from data.
    Sub in all missing values with a new line, and return this as
    a pandas dataframe.
    """
    df = pd.DataFrame(data)
    df = df.loc[:, [3,4]]
    df.columns = ['word', 'pos_0']
    df['word'] = df['word'].str.replace('""', '"')
    df['word'] = df['word'].str.strip()
    for col in ['word', 'pos_0']:
        df.loc[df[col].isnull(), col] = '\n'
    return df



def generate_name_maps():
    """
    Create mapping of male/female names to anonymised entities.
    
    We add other male/female names to the list if they are not found
    in the nltk.names corpus.
    """
    male_names = [name for name in names.words('male.txt')] + ['Saddam', 'Mao']
    female_names = [name for name in names.words('female.txt')] + ['Gong']
    full_names = set(male_names + female_names)
                    
    full_name_pairs = [[name, 'E'+str(i)] for i, name in enumerate(full_names)]
    
    return pd.DataFrame(full_name_pairs, columns=['word', 'entity'])

def flipped_gendered_words_map(path):
    """
    Load male/female word files from gn_glove.git file downloaded above
    and create a pd.DataFrame which maps words to their equivalents in
    the opposite gender.
    
    We note that there are words which are mapped to multiple others.
    We manually select which pairings we want (stored in `manual_additions`)
    and add this to the mapping to the deduplicated original dataframe.
    """
    male_words = []
    female_words = []
    with open(os.path.join(path, 'male_word_file.txt')) as f:
        for line in f:
            male_words.append(line.strip('\n'))    
    with open(os.path.join(path, 'female_word_file.txt')) as f:
        for line in f:
            female_words.append(line.strip('\n'))
              
    # Manually add words not in Zhao's subset
    male_words = male_words + ['kingdom']
    female_words = female_words + ['queendom']      
              
    full_mapping = [[m, w] for m, w in zip(male_words, female_words)] + \
        [[w, m] for m, w in zip(male_words, female_words)]
        
              
    full_mapping_df = pd.DataFrame(full_mapping, columns=['word', 'flipped_gender_word'])
    
    # Remove gendered pronoun words
    full_mapping_df = full_mapping_df.loc[~full_mapping_df['word'].str.contains('^he$|^she$|^her$|^his$|^him$')]
    
    # Remove all duplicated 'word' entries and manually re-add those which make most sense
    removed_words = set(
        full_mapping_df.loc[full_mapping_df['word'].duplicated(keep=False), 'word']
    )
    full_mapping_df = full_mapping_df.drop_duplicates(subset='word', keep=False)
       
    manual_additions = pd.DataFrame([
        ['bachelor', 'maiden'],
        ['bride' , 'bridegroom'],
        ['brides' , 'bridegrooms'],
        ['dude', 'chick'],
        ['dudes', 'chicks'],    
        ['gal', 'guy'],
        ['gals', 'guys'],
        ['god', 'goddess'],
        ['grooms', 'brides'],
        ['hostess', 'host'],
        ['ladies', 'gentlemen'],
        ['lady', 'gentleman'],
        ['lass', 'lad'],
        ['manservant', 'maid'],
        ['mare', 'stallion'],
        ['maternity', 'paternity'],
        ['paternity', 'maternity'],
        ['penis', 'vagina'],
        ['mistress', 'master'],
        ['nun', 'priest'],
        ['nuns', 'priests'],   
        ['priest', 'priestess'],  
        ['priests', 'priestesses'],  
        ['prostatic_utricle', 'womb'],
        ['sir', 'madam'],
        ['wife', 'husband']
    ], columns=['word', 'flipped_gender_word'])
    
    # Ensure all duplicated words are accounted for
    assert set(manual_additions['word']) == removed_words
    
    full_mapping_df = pd.concat([full_mapping_df, manual_additions], axis=0)
    
    return full_mapping_df



def unify_full_string_cols(d):
    """
    Unify all anonymised entities, gender flipped words and ungendered
    words into a single column.
    """
    d['original_str'] = d['word']
    d.loc[d['entity'].notnull(), 'original_str'] = d['entity']
    d.loc[d['orig_pronoun'].notnull(), 'original_str'] = d['orig_pronoun']
    
    d['flipped_str'] = d['word']
    d.loc[d['flipped_entity'].notnull(), 'flipped_str'] = d['flipped_entity']
    d.loc[d['flipped_pronoun'].notnull(), 'flipped_str'] = d['flipped_pronoun']
    d.loc[d['flipped_gender_word'].notnull(), 'flipped_str'] = d['flipped_gender_word']

    return d


def process_ontonotes_file(path):
    """
    Process ontonotes file located by `path` and process the file as a dataframe.
    Flip gendered words and anonymise entities. Concatenate all words in 
    `orignal_str` and `flipped_st` to create full original and flipped 
    strings which are returned as an output.
    """
    data = load_data(path)
    
    df = preprocess_content(data)
    pronoun_map = generate_pronoun_map()
    name_map = generate_name_maps()
    flipped_map = flipped_gendered_words_map('glove/wordlist/')
    df_2 = pd.merge(df, name_map, on='word', how='left')

    df_2['word'] = df_2['word'].str.lower()
    df_2['flipped_entity'] = 'FL_' + df_2['entity'].str[1:]
        
    df_2['orig_pronoun'] = '[' + df_2.loc[
        (df_2.loc[:, 'pos_0'].astype(str).str.contains('PRP')) &
        (df_2.loc[:, 'word'].astype(str).str.contains('^he$|^she$|^her$|^his$|^him$')),
        'word'
    ].astype(str) + ']'
    
    df_3 = pd.merge(df_2, pronoun_map, on=['word', 'pos_0'], how='left')

    df_4 = pd.merge(df_3, flipped_map, on='word', how='left')
    
    df_5 = unify_full_string_cols(df_4)

    original_string = df_5['original_str'].str.cat(sep=' ')
    flipped_string = df_5['flipped_str'].str.cat(sep=' ')    
    
    return original_string, flipped_string

PATH = "OntoNotes-5.0\\conll-formatted-ontonotes-5.0\\"
EXT = "*.csv"
all_csv_files = [file
                 for path, subdir, files in os.walk(PATH)
                 for file in glob(os.path.join(path, EXT))]
original_strings = []
flipped_strings = []
erroneous_paths = []

for path in tqdm(all_csv_files):
    try:
        original, flipped = process_ontonotes_file(path)
        original_strings.append(original)
        flipped_strings.append(flipped)
    except:
        erroneous_paths.append(path)
import re

def compile_pronoun_strings(corpus):
    """
    Note that BERT does not process strings longer than 512 characters. Thus
    we ensure that all strings are below this character limit.
    
    We attempt to add as many sentences as possible to the training example
    to provide maximal context to the BERT masked language model. We also
    add `[CLS]` and `[SEP]` tokens to our training strings.
    """
    stored_full_strings = []
    temp_storage = []
    pronouns = ['[his]', '[her]', '[him]', '[she]', '[he]']
    pronoun_regex = '\[his\]|\[her\]|\[him\]|\[she\]|\[he\]|\[hers\]'
    
    num_corpi_too_long = 0
    for subset in corpus:
        temp = subset.split('\n')
        for string in temp:
            temp_storage.append(string)
            if re.search(pronoun_regex, string):
                if len(' [SEP] '.join(temp_storage)) <= 512:
                    stored_full_strings.append('[CLS] '+' [SEP] '.join(temp_storage) + ' [SEP]')
                    temp_storage = []
                elif len(' [SEP] '.join(temp_storage[-8:])) <= 512:
                    stored_full_strings.append('[CLS] '+' [SEP] '.join(temp_storage[-8:]) + ' [SEP]')
                    temp_storage = []
                elif len(' [SEP] '.join(temp_storage[-7:])) <= 512:
                    stored_full_strings.append('[CLS] '+' [SEP] '.join(temp_storage[-7:]) + ' [SEP]')
                    temp_storage = []
                elif len(' [SEP] '.join(temp_storage[-6:])) <= 512:
                    stored_full_strings.append('[CLS] '+' [SEP] '.join(temp_storage[-6:]) + ' [SEP]')
                    temp_storage = []
                elif len(' [SEP] '.join(temp_storage[-5:])) <= 512:
                    stored_full_strings.append('[CLS] '+' [SEP] '.join(temp_storage[-5:]) + ' [SEP]')
                    temp_storage = []
                elif len(' [SEP] '.join(temp_storage[-4:])) <= 512:
                    stored_full_strings.append('[CLS] '+' [SEP] '.join(temp_storage[-4:]) + ' [SEP]')
                    temp_storage = []
                elif len(' [SEP] '.join(temp_storage[-3:])) <= 512:
                    stored_full_strings.append('[CLS] '+' [SEP] '.join(temp_storage[-3:]) + ' [SEP]')
                    temp_storage = []
                elif len(' [SEP] '.join(temp_storage[-2:])) <= 512:
                    stored_full_strings.append('[CLS] '+' [SEP] '.join(temp_storage[-2:]) + ' [SEP]')
                    temp_storage = []
                elif len(' [SEP] '.join(temp_storage[-1:])) <= 512:
                    stored_full_strings.append('[CLS] '+' [SEP] '.join(temp_storage[-1:]) + ' [SEP]')
                    temp_storage = []
                else:
                    num_corpi_too_long += 1
                    stored_full_strings.append('___TEXT-TO-LONG___')
                    temp_storage = []
    print("Number of text corpuses which are too long for BERT: {} / {}".format(num_corpi_too_long, len(corpus)))
    return stored_full_strings

original_pronoun_strings = compile_pronoun_strings(original_strings)
flipped_pronoun_strings = compile_pronoun_strings(flipped_strings)

print(len(original_pronoun_strings))
print(len(flipped_pronoun_strings))
assert len(original_pronoun_strings) == len(flipped_pronoun_strings)

orig = [s for strings in original_strings for s in strings.split('\n')]
flip = [s for strings in flipped_strings for s in strings.split('\n')]

print(len(orig))
print(len(flip))
assert len(orig) == len(flip)



def generate_training_data(strings):
    """
    Identify whether a string contains a gendered pronoun.
    If so, identify if a pronoun is missing, and replace its
    occurance with `[MASK]`, whilst keeping the pronoun as the
    predictive target.
    
    If a string has multiple pronouns, we create as many training
    examples as there are unique pronouns in the sentence.
    """
    data_list = []
    regex = '\[his\]|\[her\]|\[him\]|\[she\]|\[he\]|\[hers\]'


    for string in strings:
        string_pronouns = re.findall(regex, string)
        if string_pronouns:
            for pronoun in string_pronouns:
                regex_pronoun = re.compile('\[' + pronoun + '\]')
                temp_str = re.sub(regex, '[MASK]', string)
                temp_str = re.sub(r'\s+', r' ', temp_str)
                temp_data = [temp_str, pronoun[1:-1]]
                data_list.append(temp_data)
        else:
            pass   # Pass if no string is present
    return data_list


original_data = generate_training_data(original_pronoun_strings)
flipped_data = generate_training_data(flipped_pronoun_strings)

assert len(original_data) == len(flipped_data)

original_df = pd.DataFrame(original_data, columns=['text', 'pronouns'])
flipped_df = pd.DataFrame(flipped_data, columns=['text', 'pronouns'])

original_dropped_df = original_df.drop_duplicates(keep='first')
flipped_dropped_df = flipped_df.drop_duplicates(keep='first')

intersection_ind = flipped_dropped_df.index.intersection(original_dropped_df.index)

original_dropped_df = original_dropped_df.loc[intersection_ind]
flipped_dropped_df = flipped_dropped_df.loc[intersection_ind]

assert original_dropped_df.shape == flipped_dropped_df.shape
print(flipped_dropped_df.shape)

original_dropped_df.to_csv('original_data.csv', index=False)
flipped_dropped_df.to_csv('flipped_data.csv', index=False)


