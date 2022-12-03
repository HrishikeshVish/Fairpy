import os
import pandas as pd
from transformers import BertForMaskedLM, BertTokenizer
from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup
from keras.utils import pad_sequences
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from nltk import sent_tokenize
from techniques.genderAugmentRetrain.becpro_utils import input_pipeline, mask_tokens
from techniques.genderAugmentRetrain.Augment_utils import counter_factual_augmentation
import time
import datetime
import random
import sys
from tqdm import tqdm
import nltk
nltk.download('punkt')
import math

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def save_model(processed_model, tokenizer, epoch, lr, eps):
  output_dir = 'model_save/debias/full/lr_{}_eps_{}/epoch_{}/'.format(lr, eps, epoch)

  # Create output directory if needed
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)

  print("Saving model to %s" % output_dir)

  # Save a trained model, configuration and tokenizer using `save_pretrained()`.
  # They can then be reloaded using `from_pretrained()`
  model_to_save = processed_model.module if hasattr(processed_model, 'module') else processed_model  # Take care of distributed/parallel training
  model_to_save.save_pretrained(output_dir)
  tokenizer.save_pretrained(output_dir)

  # Good practice: save your training arguments together with the trained model
  torch.save([epoch, lr, eps], os.path.join(output_dir, 'training_args.bin'))

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    labels_flat_filtered = (labels_flat != 0) * (labels_flat != -100) * labels_flat

    return np.sum((pred_flat == labels_flat_filtered) * (labels_flat_filtered != 0)) / sum(labels_flat_filtered != 0)

def preprocess(tokenizer):
    try:
        input_ids = np.load(sys.path[2]+'data/input.npy')
        masked_lm_labels = np.load(sys.path[2]+'data/masked_lm.npy')
        attention_masks = np.load(sys.path[2]+'data/attention.npy')
        return input_ids, masked_lm_labels, attention_masks
    except:
        df_orig = pd.read_csv(sys.path[2]+'data/original_data.csv')
        df_flipped = pd.read_csv(sys.path[2]+'data/flipped_data.csv')
        df = pd.concat([df_orig, df_flipped])
        df['gender'] = df['pronouns'].str.contains('^he$|^his$|^him$').astype(int)
        # Report the number of sentences.
        #print('Number of training sentences: {:,}\n'.format(df.shape[0]))
        #print(df.sample(10))
        #print(df.gender.value_counts())
        sentences = df.text.values
        labels = df.pronouns.values
        #print(' Original: ', sentences[0])
        #print('Tokenized: ', tokenizer.tokenize(sentences[0]))
        #print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))
        mask_id = tokenizer.convert_tokens_to_ids('[MASK]')
        print(mask_id)
        masked_lm_labels = []
        counter = 0
        input_ids = []
        for sentence, label in zip(sentences, labels):
            sentence_ids = tokenizer.encode(sentence)
            #print(sentence_ids)
            label_id = tokenizer.convert_tokens_to_ids(label)
            masked_lm_labels.append([label_id if id == mask_id else -100 for id in sentence_ids])
            input_ids.append(sentence_ids)
        word_count = [len(s.split()) for s in list(sentences)]
        print(sum(word_count))
        
        #for sent in list(sentences):
        #    encoded_sent = tokenizer.encode(sent, add_special_tokens=False),
        #    input_ids.append(encoded_sent)
        #print('Original: ', sentences[0])
        #print('Token IDs: ', input_ids[0])
        print('Max Sentence Length: ', max([len(sen) for sen in input_ids]))
        MAX_LEN = 164
        print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)
        print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))
        input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
        masked_lm_labels = pad_sequences(masked_lm_labels, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
        print('\nDone.')
        attention_masks = []
        for sent in input_ids:
            att_mask = [int(token_id > 0) for token_id in sent]
            attention_masks.append(att_mask)
        input_ids = np.asarray(input_ids)
        masked_lm_labels = np.asarray(masked_lm_labels)
        attention_masks = np.asarray(attention_masks)
        with open(sys.path[2]+'data/input.npy', 'wb') as f:
            np.save(f, input_ids)
        with open(sys.path[2]+'data/masked_lm.npy', 'wb') as f:
            np.save(f, masked_lm_labels)
        with open(sys.path[2]+'data/attention.npy', 'wb') as f:
            np.save(f, attention_masks)
        return input_ids, masked_lm_labels, attention_masks
def load_cnn_data(tokenizer):
    input_ids, masked_lm_labels, attention_masks = preprocess(tokenizer)
    train_inputs, validation_inputs, train_lm_labels, validation_lm_labels = train_test_split(input_ids, masked_lm_labels,
                                                            random_state=2018, test_size=0.2)
    # Do the same for the masks.
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, masked_lm_labels,random_state=2018, test_size=0.2)
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)

    train_lm_labels = torch.tensor(train_lm_labels)
    validation_lm_labels = torch.tensor(validation_lm_labels)

    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    batch_size = 16

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_lm_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_lm_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    return train_dataloader, validation_dataloader
def load_becpro_data(tokenizer):
    eval_data = pd.read_csv(sys.path[2]+'data/bec-pro/EEC_TM_TAM.tsv', sep='\t')
    tune_corpus = pd.read_csv('data/bec-pro/gap_flipped.tsv', sep='\t')
    tune_data = []
    for text in tune_corpus.Text:
        tune_data += sent_tokenize(text)
    max_len_tune = max([len(sent.split()) for sent in tune_data])
    pos = math.ceil(math.log2(max_len_tune))
    max_len_tune = int(math.pow(2, pos))
    tune_tokens, tune_attentions = input_pipeline(tune_data, tokenizer, max_len_tune)
    assert tune_tokens.shape == tune_attentions.shape
    batch_size = 1
    train_data = TensorDataset(tune_tokens, tune_attentions)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size=batch_size)

    return train_dataloader
def load_text_data(file, tokenizer, bias_type='gender'):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    tune_data = []
    lines = counter_factual_augmentation(lines, bias_type)
    for line in lines:
        tune_data += sent_tokenize(line)
    max_len_tune = max([len(sent.split()) for sent in tune_data])
    pos = math.ceil(math.log2(max_len_tune))
    max_len_tune = int(math.pow(2, pos))
    tune_tokens, tune_attentions = input_pipeline(tune_data, tokenizer, max_len_tune)
    assert tune_tokens.shape == tune_attentions.shape
    batch_size = 1
    train_data = TensorDataset(tune_tokens, tune_attentions)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size=batch_size)
    return train_dataloader

def fineTune(device, model, tokenizer, dataset_name, dataset_loc='', bias_type='gender'):
    if(dataset_name == 'cnn'): 
        train_dataloader, validation_dataloader = load_cnn_data(tokenizer)
    elif('bec' in dataset_name):
        train_dataloader = load_becpro_data(tokenizer)
    else:
        train_dataloader = load_text_data(dataset_loc,tokenizer, bias_type)
        
    model.cuda()

    lr = 2e-5
    eps = 1e-8
    optimizer = AdamW(model.parameters(),
                  lr = lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = eps # args.adam_epsilon  - default is 1e-8.
                )
    epochs = 8
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
    
    
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    training_loss_values=[]
    eval_loss_values = []
    for epoch_i in range(2, epochs):

        # Measure how long the training epoch takes.
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 20 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                print('Batch {:>5,} of {:>5,}.  Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            #   [3]: segments
            if(dataset_name == 'cnn'):
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].type(torch.LongTensor).to(device)
            else:
                b_input_ids, b_labels = mask_tokens(batch[0].type(torch.LongTensor), tokenizer)
                b_input_ids = b_input_ids.to(device)
                b_labels = b_labels.to(device)
                b_input_mask = batch[1].to(device)


            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids, 
                        # token_type_ids=b_segments,
                        attention_mask=b_input_mask, 
                        labels=b_labels)
            
            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple.
            loss = outputs[0]

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)            
        
        # Store the loss value for plotting the learning curve.
        training_loss_values.append(avg_train_loss)
        #save_model(model, epoch_i, lr, eps)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        if(dataset_name == 'cnn'):
                
            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            # Tracking variables 
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                
                # Add batch to GPU
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                
                # Telling the model not to compute or store gradients, saving memory and
                # speeding up validation
                with torch.no_grad():        

                    # Forward pass, calculate logit predictions.
                    # This will return the logits rather than the loss because we have
                    # not provided labels.
                    # token_type_ids is the same as the "segment ids", which 
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here: 
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    outputs = model(b_input_ids, 
                            # token_type_ids=b_segments,
                            attention_mask=b_input_mask,
                            labels=b_labels)
                
                # Get testing loss
                loss = outputs[0]
                eval_loss += loss.item()

                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                logits = outputs[1]        

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                
                # Calculate the accuracy for this batch of test sentences.
                tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                
                # Accumulate the total accuracy.
                eval_accuracy += tmp_eval_accuracy

                # Track the number of batches
                nb_eval_steps += 1

            # Calculate the average loss over the training data.
            avg_eval_loss = eval_loss / len(validation_dataloader)            
            eval_loss_values.append(avg_eval_loss)


            # Report the final accuracy for this validation run.
            print("  Average evaluation loss: {0:.2f}".format(avg_eval_loss))
            print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
            print("  Validation took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("Training complete!")

    return model

