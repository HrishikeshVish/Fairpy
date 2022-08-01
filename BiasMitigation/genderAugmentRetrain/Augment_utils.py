from audioop import bias
import numpy as np
import pandas as pd
import json
import random
attribute_file = f"data/bias_attribute_words.json"
def _create_bias_attribute_words(bias_type):
    """Creates list of bias attribute words (e.g., he/she).

    Args:
        attribute_file: Path to the file containing the bias attribute words.
        bias_type: Type of bias attribute words to load. Must be one of
            ["gender", "race", "religion"].

    Notes:
        * We combine each bias attribute word with several punctuation marks.
            The current set of words is *not* exhaustive, however, it should
            cover most occurances.
    """
    with open(attribute_file, "r") as f:
        bias_attribute_words = json.load(f)[bias_type]

    result = bias_attribute_words[:]
    for punctuation in [".", ",", "?", "!", ";", ":"]:
        for words in bias_attribute_words:
            augmented_words = [word + punctuation for word in words]
            result.append(augmented_words)
    #print(result)
    return result
#_create_bias_attribute_words(attribute_file, 'gender')

def gender_counterfactual_augmentation(examples):
    """Applies gender counterfactual data augmentation to a batch of examples.

    Notes:
        * We apply CDA after the examples have potentially been grouped.
        * This implementation can be made more efficient by operating on
            token IDs as opposed to text. We currently decode each example
            as it is simpler.
    """
    bias_attribute_words = _create_bias_attribute_words('gender')
    outputs = []
    for input_ids in examples:
        # For simplicity, decode each example. It is easier to apply augmentation
        # on text as opposed to token IDs.
        sentence = input_ids.lower()
        words = sentence.split()  # Tokenize based on whitespace.
        augmented_sentence = words[:]

        augmented = False
        for position, word in enumerate(words):
            for male_word, female_word in bias_attribute_words:
                if male_word == word:
                    augmented = True
                    augmented_sentence[position] = female_word

                if female_word == word:
                    augmented = True
                    augmented_sentence[position] = male_word

        if augmented:
            augmented_sentence = " ".join(augmented_sentence)
            outputs.append(augmented_sentence)
            outputs.append(sentence)

    # There are potentially no counterfactual examples.
    if not outputs:
        return {"input_ids": [], "attention_mask": []}

    """return tokenizer(
        outputs,
        return_special_tokens_mask=True,
        add_special_tokens=False,  # Special tokens are already added.
        truncation=True,
        padding=True,
    )"""

def ternary_counterfactual_augmentation(examples, bias_type):
    """Applies racial/religious counterfactual data augmentation to a batch of
    examples.

    Notes:
        * We apply CDA after the examples have potentially been grouped.
        * This implementation can be made more efficient by operating on
            token IDs as opposed to text. We currently decode each example
            as it is simpler.
    """
    bias_attribute_words = _create_bias_attribute_words(bias_type)
    outputs = []
    for input_ids in examples:
        # For simplicity, decode each example. It is easier to apply augmentation
        # on text as opposed to token IDs.
        sentence = input_ids.lower()
        words = sentence.split()  # Tokenize based on whitespace.
        augmented_sentence = words[:]

        # Sample the augmentation pairs.
        r1_augmentation_pair = random.choice([1, 2])
        r2_augmentation_pair = random.choice([0, 2])
        r3_augmentation_pair = random.choice([0, 1])

        augmented = False
        for position, word in enumerate(words):
            for augmentation_words in bias_attribute_words:
                # Implementation here.
                r1_word, r2_word, r3_word = augmentation_words

                if r1_word == word:
                    augmented = True
                    augmented_sentence[position] = augmentation_words[
                        r1_augmentation_pair
                    ]

                if r2_word == word:
                    augmented = True
                    augmented_sentence[position] = augmentation_words[
                        r2_augmentation_pair
                    ]

                if r3_word == word:
                    augmented = True
                    augmented_sentence[position] = augmentation_words[
                        r3_augmentation_pair
                    ]

        if augmented:
            augmented_sentence = " ".join(augmented_sentence)
            outputs.append(augmented_sentence)
            outputs.append(sentence)

    # There are potentially no counterfactual examples.
    if not outputs:
        return {"input_ids": [], "attention_mask": []}

def quarternary_counterfactual_augmentation(examples, bias_type):
    """Applies racial/religious counterfactual data augmentation to a batch of
    examples.

    Notes:
        * We apply CDA after the examples have potentially been grouped.
        * This implementation can be made more efficient by operating on
            token IDs as opposed to text. We currently decode each example
            as it is simpler.
    """
    bias_attribute_words = _create_bias_attribute_words(bias_type)
    outputs = []
    for input_ids in examples:
        # For simplicity, decode each example. It is easier to apply augmentation
        # on text as opposed to token IDs.
        sentence = input_ids.lower()
        words = sentence.split()  # Tokenize based on whitespace.
        augmented_sentence = words[:]

        # Sample the augmentation pairs.
        r1_augmentation_pair = random.choice([1, 2, 3])
        r2_augmentation_pair = random.choice([0, 2, 3])
        r3_augmentation_pair = random.choice([0, 1, 3])
        r4_augmentation_pair = random.choice([0, 1, 2])
        augmented = False
        for position, word in enumerate(words):
            for augmentation_words in bias_attribute_words:
                # Implementation here.
                r1_word, r2_word, r3_word, r4_word = augmentation_words

                if r1_word == word:
                    augmented = True
                    augmented_sentence[position] = augmentation_words[
                        r1_augmentation_pair
                    ]

                if r2_word == word:
                    augmented = True
                    augmented_sentence[position] = augmentation_words[
                        r2_augmentation_pair
                    ]

                if r3_word == word:
                    augmented = True
                    augmented_sentence[position] = augmentation_words[
                        r3_augmentation_pair
                    ]
                if r4_word == word:
                    augmented = True
                    augmented_sentence[position] = augmentation_words[
                        r4_augmentation_pair
                    ]

        if augmented:
            augmented_sentence = " ".join(augmented_sentence)
            outputs.append(augmented_sentence)
            outputs.append(sentence)

    # There are potentially no counterfactual examples.
    if not outputs:
        return {"input_ids": [], "attention_mask": []}
def non_binary_counterfactual_augmentation(examples):
    bias_attribute_words = _create_bias_attribute_words('non-binary')
    outputs = []
    for input_ids in examples:
        # For simplicity, decode each example. It is easier to apply augmentation
        # on text as opposed to token IDs.
        sentence = input_ids.lower()
        words = sentence.split()  # Tokenize based on whitespace.
        augmented_sentence = words[:]

        # Sample the augmentation pairs.
        augment_choice = random.choice(range(1, 9))

        augmented = False
        for position, word in enumerate(words):
            for augmentation_words in bias_attribute_words:
                # Implementation here.
                dominant_word = augmentation_words[0]

                if dominant_word == word:
                    augmented = True
                    augmented_sentence[position] = augmentation_words[
                        augment_choice
                    ]

        if augmented:
            augmented_sentence = " ".join(augmented_sentence)
            outputs.append(augmented_sentence)
            outputs.append(sentence)

    # There are potentially no counterfactual examples.
    if not outputs:
        return {"input_ids": [], "attention_mask": []}

def ethnicity_counterfactual_augmentation(examples):
    bias_attribute_words = _create_bias_attribute_words('ethnicity')
    ethnicity_size = len(bias_attribute_words[0])
    choice_options = list(range(ethnicity_size))
    random_choices = np.random.permutation(choice_options)
    
    outputs = []
    for input_ids in examples:
        # For simplicity, decode each example. It is easier to apply augmentation
        # on text as opposed to token IDs.
        sentence = input_ids.lower()
        words = sentence.split()  # Tokenize based on whitespace.
        augmented_sentence = words[:]

        # Sample the augmentation pairs.
        augment_choice = random.choice(range(1, 9))

        augmented = False
        for position, word in enumerate(words):
            for augmentation_words in bias_attribute_words:
                # Implementation here.
                if(word in augmentation_words):
                    augmented = True
                    index = augmentation_words.index(word)
                    augment_choice = random_choices[index]
                    augmented_sentence[position] = augmentation_words[augment_choice]

        if augmented:
            augmented_sentence = " ".join(augmented_sentence)
            outputs.append(augmented_sentence)
            outputs.append(sentence)

    # There are potentially no counterfactual examples.
    if not outputs:
        return {"input_ids": [], "attention_mask": []}
