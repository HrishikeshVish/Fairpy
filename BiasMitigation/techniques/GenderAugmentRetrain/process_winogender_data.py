import csv

TEMPLATES_PATH = 'winogender-schemas/data/templates.tsv'
OCC_STATS_PATH = 'winogender-schemas/data/occupations-stats.tsv'
OUTPUT_PATH = 'processed_wino_data.txt'
PRONOUNS = {
    '$NOM_PRONOUN': {'M': '[he]', 'F': '[she]', 'N': '[they]'},
    '$ACC_PRONOUN': {'M': '[him]', 'F': '[her]', 'N': '[them]'},
    '$POSS_PRONOUN': {'M': '[his]', 'F': '[her]', 'N': '[their]'}
}


def process_occ_stats(occ_data):
    occ_bias = {}
    for row in occ_data:
        occupation = row['occupation']
        bergsma_pct_f = float(row['bergsma_pct_female'])
        if bergsma_pct_f > 50:
            stereo = 'F'
        else:
            stereo = 'M'
        occ_bias[occupation] = stereo
    return occ_bias


def process_wino_data(wino_data, occ_bias):
    processed_wino_data = []
    for wino_dict in wino_data:
        occupation = wino_dict['occupation(0)']
        participant = wino_dict['other-participant(1)']
        sentence = wino_dict['sentence']
        answer = int(wino_dict['answer'])
    
        if answer == 0:
            bias_gender = occ_bias[occupation]
            occupation = '[' + occupation + ']'
    
            sentence = sentence.replace('$OCCUPATION', occupation)
            sentence = sentence.replace('$PARTICIPANT', participant)
            
            for k, v in PRONOUNS.items():
                if k in sentence:
                    sentence = sentence.replace(k, PRONOUNS[k][bias_gender])
            processed_wino_data.append(sentence)
    return processed_wino_data


def main():
    templates_data = []
    occ_stats_data = []
    with open(TEMPLATES_PATH) as f:
        read_tsv = csv.DictReader(f, delimiter='\t')
        for line in read_tsv:
            templates_data.append(line)

    with open(OCC_STATS_PATH) as f:
        read_tsv = csv.DictReader(f, delimiter='\t')
        for line in read_tsv:
            occ_stats_data.append(line)

    occ_bias = process_occ_stats(occ_stats_data) 
    wino_processed = process_wino_data(templates_data, occ_bias) 

    with open(OUTPUT_PATH, 'w') as f:
        for item in wino_processed:
            f.write(f"{item} \n")

if __name__ == '__main__':
    main()
