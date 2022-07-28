//  Modifications copyright (C) 2013 <University of California, Los Angeles/uclannlp>
//  GloVe: Global Vectors for Word Representation
//
//  Copyright (c) 2014 The Board of Trustees of
//  The Leland Stanford Junior University. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//
//  For more information, bug reports, fixes, contact:
//    Jeffrey Pennington (jpennin@stanford.edu)
//    GlobalVectors@googlegroups.com
//    http://nlp.stanford.edu/projects/glove/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <assert.h>
#include "const.h"
#include "cooccur.h"
#include "data_struct.h"
#include "get_vocabhash.h"

//typedef struct hashrec {
//    char        *word;
//    long long id;
//    struct hashrec *next;
//} HASHREC;


real lambda1 = 0.8;
real lambda2 = 0.8;
int verbose = 2; // 0, 1, or 2
int use_unk_vec = 1; // 0 or 1
int num_threads = 8; // pthreads
int num_iter = 25; // Number of full passes through cooccurrence matrix
int vector_size = 50; // Word vector size
int save_gradsq = 0; // By default don't save squared gradient values
int use_binary = 1; // 0: save as text files; 1: save as binary; 2: both. For binary, save both word and context word vectors.
int model = 2; // For text file output only. 0: concatenate word and context vectors (and biases) i.e. save everything; 1: Just save word vectors (no bias); 2: Save (word + context word) vectors (no biases)
real eta = 0.05; // Initial learning rate
real alpha = 0.75, x_max = 100.0; // Weighting function parameters, not extremely sensitive to corpus, though may need adjustment for very small or very large corpora

real *W, *gradsq, *cost;
long long num_lines, *lines_per_thread, vocab_size;
char *vocab_file, *input_file, *save_W_file, *save_gradsq_file;

// New variables for debias task
char **male_words, **female_words;
real *gender_direction, *temp_vector;
int *male_index, *female_index;
char *male_word_file, *female_word_file, *vocab_hash_file;
long long pair_count;
int male_count=0, female_count = 0;

HASHREC **vocab_hash;



/* Initialize parameters */
void initialize_parameters() {
	long long a, b;
	vector_size++; // Temporarily increment to allocate space for bias

	/* Allocate space for word vectors and context word vectors, and correspodning gradsq */
	a = posix_memalign((void **)&W, 128, 2 * vocab_size * vector_size * sizeof(real)); // Might perform better than malloc
    if (W == NULL) {
        fprintf(stderr, "Error allocating memory for W\n");
        exit(1);
    }
    a = posix_memalign((void **)&gradsq, 128, 2 * vocab_size * vector_size * sizeof(real)); // Might perform better than malloc
	if (gradsq == NULL) {
        fprintf(stderr, "Error allocating memory for gradsq\n");
        exit(1);
    }
	for (b = 0; b < vector_size; b++)
        for (a = 0; a < 2 * vocab_size; a++)
            W[a * vector_size + b] = ( rand() / (real)RAND_MAX - 0.5 ) / vector_size;
	for (b = 0; b < vector_size; b++)
        for (a = 0; a < 2 * vocab_size; a++)
            gradsq[a * vector_size + b] = 1.0; // So initial value of eta is equal to initial learning rate
	vector_size--;


    /* Read the gender_word list files; maintain the gender_pairs and neutral word list for later*/

    FILE *gin;
    gin = fopen(male_word_file, "r");
    pair_count = 0;
    char *gender_word, *ptr;
    gender_word = malloc(sizeof(char) * 100);
    male_words = (char **)malloc(sizeof(char *)*MAX_WORD_COUNT);
    female_words = (char **)malloc(sizeof(char *)*MAX_WORD_COUNT);
    while(fscanf(gin, "%s", gender_word) != EOF){
        if((ptr = strchr(gender_word, '\n')) != NULL)  *ptr = '\0';
        //male_words[pair_count] = (char *)malloc(MAX_WORD_LENGTH * sizeof(char));
        //strcpy(male_words[pair_count], gender_word);
        male_words[pair_count] = (char *)gender_word;
        gender_word = malloc(sizeof(char) * 100);
        pair_count++;
    }
    fclose(gin);
    fprintf(stderr, "%lld male words in the gender word list\n", pair_count);

    int male_count_tmp = pair_count;
    gin = fopen(female_word_file, "r");
    pair_count = 0;
    while(fscanf(gin, "%s", gender_word) != EOF) {
        if ((ptr = strchr(gender_word, '\n')) != NULL) *ptr = '\0';
        female_words[pair_count] = (char *) gender_word;
        gender_word = malloc(sizeof(char) * 100);
        pair_count++;
    }
    fclose(gin);
    fprintf(stderr, "%lld female words in the gender word list\n", pair_count);
    assert(male_count_tmp == pair_count);


    /* Get the index for those gender specific words*/
    int i, word_index;

    //(Finished)  load vocab_hash from file
    //  copy/paste hashsearch function here

    /* Initialze word vector for gender specific words*/
    HASHREC *temp;
    vocab_hash = get_vocabhash(vocab_hash_file);
    male_index = malloc(sizeof(int) * MAX_WORD_COUNT);
    female_index = malloc(sizeof(int) * MAX_WORD_COUNT);
    male_count = 0;
    female_count = 0;
    for (i = 0; i < pair_count ; i++){
        temp = hashsearch(vocab_hash, male_words[i]);
        
        if (temp != NULL){
            male_count ++;
            word_index = temp->id;
            W[word_index * (vector_size + 1) - 2] = 1; // TODO: check this is correct
            male_index[i] = word_index;
        }

        temp = hashsearch(vocab_hash, female_words[i]);
        if (temp!=NULL){
            female_count++;
            word_index = temp -> id;
	    //fprintf(stderr, "%s\n", temp->word);
            W[word_index * (vector_size + 1) - 2] = -1; // TODO: check this is correct
            female_index[i]  = word_index;
	    fprintf(stderr, "%s ", female_words[i]);
        }
    }
    fprintf(stderr, "%d male words in vocab\n", male_count);
    fprintf(stderr, "%d female words in vocab\n", female_count);

}

/* Check whether a word with index `index` is a gender word.
 *      1  for male
 *     -1  for female
 *      0  for neutral */
int get_gender(long long index){
    int a;
    for(a = 0; a < pair_count; a++){
        if (male_index[a] == index)
            return 1;
        if (female_index[a] == index)
            return -1;
    }
    return 0;
}


/* Train the GloVe model */
void *glove_thread(void *vid) {
    long long a, b, c, l1, l2;
    long long id = (long long) vid;
    CREC cr;
    real diff, fdiff, temp1, temp2, temp, diff1, diff2;
    FILE *fin;
    fin = fopen(input_file, "rb");
    fseeko(fin, (num_lines / num_threads * id) * (sizeof(CREC)), SEEK_SET); //Threads spaced roughly equally throughout file
    cost[id] = 0;



    for(a = 0; a < lines_per_thread[id]; a++) {
        fread(&cr, sizeof(CREC), 1, fin);
        if(feof(fin)) break;

        /* Get location of words in W & gradsq */
        l1 = (cr.word1 - 1LL) * (vector_size + 1); // cr word indices start at 1
        l2 = ((cr.word2 - 1LL) + vocab_size) * (vector_size + 1); // shift by vocab_size to get separate vectors for context words

        /* Calculate cost, save diff for gradients */
        diff = 0;
        for(b = 0; b < vector_size; b++) diff += W[b + l1] * W[b + l2]; // dot product of word and context word vector
        diff += W[vector_size + l1] + W[vector_size + l2] - log(cr.val); // add separate bias for each word
        fdiff = (cr.val > x_max) ? diff : pow(cr.val / x_max, alpha) * diff; // multiply weighting function (f) with diff
        cost[id] += 0.5 * fdiff * diff; // weighted squared error --J0

        diff1 = 0; //wg
        diff2 = 0;


        int Wi_gender = get_gender(cr.word1);
        int Wj_gender = get_gender(cr.word2);

        switch(Wi_gender) {
            case 1: //male
                diff1 += W[vector_size + l1 - 1] / male_count;
                break;
            case -1: //female
                diff1 -= W[vector_size + l1 - 1] / female_count;
                break;
            default: //neutral
                for(b = 0; b < vector_size - 1; b++) {
                    diff2 += gender_direction[b] * W[b + l1];
                }
                break;
        }
        switch(Wj_gender) {
            case 1: //male
                diff1 += W[vector_size + l2 - 1] / male_count;
                break;
            case -1: //female
                diff1 -= W[vector_size + l2 - 1] / female_count;
                break;
            default: //neutral
                for(b = 0; b < vector_size - 1; b++) {
                    diff2 += gender_direction[b] * W[b + l2];
                }
                break;
        }


        cost[id] = cost[id] - lambda1 * diff1 + lambda2 * diff2;


        /* Adaptive gradient updates */
        fdiff *= eta; // for ease in calculating gradient with responding to J0
        for(b = 0; b < vector_size; b++) {
            // learning rate times gradient for word vectors
            temp1 = fdiff * W[b + l2];
            temp2 = fdiff * W[b + l1];
            // adaptive updates
            //W[b + l1] -= temp1 / sqrt(gradsq[b + l1]);
            //W[b + l2] -= temp2 / sqrt(gradsq[b + l2]);
            W[b + l1] -= temp1 /1;
            W[b + l2] -= temp2 / 1;
            gradsq[b + l1] += temp1 * temp1;
            gradsq[b + l2] += temp2 * temp2;
        }
        // updates for bias terms
        //W[vector_size + l1] -= fdiff / sqrt(gradsq[vector_size + l1]);
        //W[vector_size + l2] -= fdiff / sqrt(gradsq[vector_size + l2]);
        W[vector_size + l1] -= fdiff /1;
        W[vector_size + l2] -= fdiff / 1;
        fdiff *= fdiff;
        gradsq[vector_size + l1] += fdiff;
        gradsq[vector_size + l2] += fdiff; //---Finish J0


        temp = 0;
        switch(Wi_gender) {
            case 1: //male
                W[vector_size + l1 - 1] += eta * lambda1;
                /* when udpating for male or female words, we ignore the gradient from J2 -- at this time, we think
                 * the gender direction is approximately fixed. */
                //for(c = 0; c < neutral_word_size; c++) {
                //	temp = 0;
                //	for(b = 0; b < vector_size - 1; b++) {
                //		temp_vector[b] = 0;
                //	}
	            //    for(b = 0; b < vector_size - 1; b++) {
	            //        temp += 2 * lambda2 * gender_direction[b] * neutral_word[c][b];
	            //    }
	            //    for(b = 0; b < vector_size - 1; b++) {
	            //        neutral_word[c][b] *= temp;
	            //        temp_vector[b] += neutral_word[c][b];
	            //    }
	            //}
	            //for(b = 0; b < vector_size - 1; b++) {
	            //    W[l1 + b] -= temp_vector[b];
	            //}
                break;
            case -1: //female
                W[vector_size + l1 - 1] -= eta * lambda1;

                break;
            case 0: //neutral
                for(b = 0; b < vector_size - 1; b++){
                    temp += 2 * lambda2 * W[l1 + b] * gender_direction[b];
                }
                for(b = 0; b < vector_size - 1; b++){
                    W[l1 + b] -= eta * temp * gender_direction[b];
                }
                break;
        }
        for(b = 0; b < vector_size; b++){
            if(W[l1 + b] > 1)
                W[l1 + b] = 1;
            if (W[l1 + b] < -1)
                W[l1 + b] = -1;
        }

        temp = 0;
        switch(Wj_gender) {
            case 1: //male
                W[vector_size + l2 - 1] += eta * lambda1;
                break;
            case -1: //female
                W[vector_size + l2 - 1] -= eta * lambda1;
                break;
            case 0: //neutral
                for(b = 0; b < vector_size - 1; b++){
                    temp += 2 * lambda2 * W[l2 + b] * gender_direction[b];
                }
                for(b = 0; b < vector_size - 1; b++){
                    W[l2 + b] -= eta * temp * gender_direction[b];
		}
                break;
        }
        for(b = 0; b < vector_size; b++){
		if(W[l2 + b] > 1) W[l2 + b] = 1;
		if(W[l2 + b] < -1) W[l2 + b] = -1;
	}
    }

    fclose(fin);
    pthread_exit(NULL);
}

/* Save params to file */
int save_params() {
    long long a, b;
    char format[20];
    char output_file[MAX_STRING_LENGTH], output_file_gsq[MAX_STRING_LENGTH];
    char *word = malloc(sizeof(char) * MAX_STRING_LENGTH);
    FILE *fid, *fout, *fgs;

    if(use_binary > 0) { // Save parameters in binary file
        sprintf(output_file,"%s.bin",save_W_file);
        fout = fopen(output_file,"wb");
        if(fout == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_W_file); return 1;}
        for(a = 0; a < 2 * (long long)vocab_size * (vector_size + 1); a++) fwrite(&W[a], sizeof(real), 1,fout);
        fclose(fout);
        if(save_gradsq > 0) {
            sprintf(output_file_gsq,"%s.bin",save_gradsq_file);
            fgs = fopen(output_file_gsq,"wb");
            if(fgs == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_gradsq_file); return 1;}
            for(a = 0; a < 2 * (long long)vocab_size * (vector_size + 1); a++) fwrite(&gradsq[a], sizeof(real), 1,fgs);
            fclose(fgs);
        }
    }
    if(use_binary != 1) { // Save parameters in text file
        sprintf(output_file,"%s.txt",save_W_file);
        if(save_gradsq > 0) {
            sprintf(output_file_gsq,"%s.txt",save_gradsq_file);
            fgs = fopen(output_file_gsq,"wb");
            if(fgs == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_gradsq_file); return 1;}
        }
        fout = fopen(output_file,"wb");
        if(fout == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_W_file); return 1;}
        fid = fopen(vocab_file, "r");
        sprintf(format,"%%%ds",MAX_STRING_LENGTH);
        if(fid == NULL) {fprintf(stderr, "Unable to open file %s.\n",vocab_file); return 1;}
        for(a = 0; a < vocab_size; a++) {
            if(fscanf(fid,format,word) == 0) return 1;
            // input vocab cannot contain special <unk> keyword
            if(strcmp(word, "<unk>") == 0) return 1;
            fprintf(fout, "%s",word);
            if(model == 0) { // Save all parameters (including bias)
                for(b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b]);
                for(b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", W[(vocab_size + a) * (vector_size + 1) + b]);
            }
            if(model == 1) // Save only "word" vectors (without bias)
                for(b = 0; b < vector_size; b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b]);
            if(model == 2) // Save "word + context word" vectors (without bias)
                for(b = 0; b < vector_size; b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b] + W[(vocab_size + a) * (vector_size + 1) + b]);
            fprintf(fout,"\n");
            if(save_gradsq > 0) { // Save gradsq
                fprintf(fgs, "%s",word);
                for(b = 0; b < (vector_size + 1); b++) fprintf(fgs," %lf", gradsq[a * (vector_size + 1) + b]);
                for(b = 0; b < (vector_size + 1); b++) fprintf(fgs," %lf", gradsq[(vocab_size + a) * (vector_size + 1) + b]);
                fprintf(fgs,"\n");
            }
            if(fscanf(fid,format,word) == 0) return 1; // Eat irrelevant frequency entry
        }

        if (use_unk_vec) {
            real* unk_vec = (real*)calloc((vector_size + 1), sizeof(real));
            real* unk_context = (real*)calloc((vector_size + 1), sizeof(real));
            word = "<unk>";

            int num_rare_words = vocab_size < 100 ? vocab_size : 100;

            for(a = vocab_size - num_rare_words; a < vocab_size; a++) {
                for(b = 0; b < (vector_size + 1); b++) {
                    unk_vec[b] += W[a * (vector_size + 1) + b] / num_rare_words;
                    unk_context[b] += W[(vocab_size + a) * (vector_size + 1) + b] / num_rare_words;
                }
            }

            fprintf(fout, "%s",word);
            if(model == 0) { // Save all parameters (including bias)
                for(b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", unk_vec[b]);
                for(b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", unk_context[b]);
            }
            if(model == 1) // Save only "word" vectors (without bias)
                for(b = 0; b < vector_size; b++) fprintf(fout," %lf", unk_vec[b]);
            if(model == 2) // Save "word + context word" vectors (without bias)
                for(b = 0; b < vector_size; b++) fprintf(fout," %lf", unk_vec[b] + unk_context[b]);
            fprintf(fout,"\n");

            free(unk_vec);
            free(unk_context);
        }

        fclose(fid);
        fclose(fout);
        if(save_gradsq > 0) fclose(fgs);
    }
    return 0;
}

/* Train model */
int train_glove() {
    long long a, file_size;
    int b;
    FILE *fin;
    real total_cost = 0;
    fprintf(stderr, "TRAINING MODEL\n");

    // Open file and get the number of lines in the file
    fin = fopen(input_file, "rb");
    if(fin == NULL) {fprintf(stderr,"Unable to open cooccurrence file %s.\n",input_file); return 1;}
    fseeko(fin, 0, SEEK_END);
    file_size = ftello(fin);
    num_lines = file_size/(sizeof(CREC)); // Assuming the file isn't corrupt and consists only of CREC's
    fclose(fin);
    fprintf(stderr,"Read %lld lines.\n", num_lines);

    if(verbose > 1) fprintf(stderr,"Initializing parameters...\n");
    initialize_parameters();
    if(verbose > 1) fprintf(stderr,"done.\n");
    if(verbose > 0) fprintf(stderr,"vector size: %d\n", vector_size);
    if(verbose > 0) fprintf(stderr,"vocab size: %lld\n", vocab_size);
    if(verbose > 0) fprintf(stderr,"x_max: %lf\n", x_max);
    if(verbose > 0) fprintf(stderr,"alpha: %lf\n", alpha);

    gender_direction = (real *)malloc(sizeof(real) * (vector_size - 1));
    memset( gender_direction, 0, sizeof(real) * (vector_size - 1));

    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    lines_per_thread = (long long *)malloc(num_threads * sizeof(long long));


    // Lock-free asynchronous SGD

    for(b = 0; b < num_iter; b++) {

        // TODO:
        //      replace b with other index traverse var
        //      figure out when gender direction is initialized

        int word1_index_1, word1_index_2;
        int c = 0;
        int gender_direction_count = 0;
        HASHREC *tmp;
        for (a = 0; a < pair_count; a++) {
            word1_index_1 = -1;
            word1_index_2 = -1;
            tmp = hashsearch(vocab_hash, male_words[a]);
            if(tmp!=NULL)
                word1_index_1 = tmp -> id;
            tmp = hashsearch(vocab_hash, female_words[a]);
            if (tmp!=NULL)
                word1_index_2 = tmp -> id;
           if(word1_index_1!=-1 && word1_index_2!= -1) {
               gender_direction_count++;
               for (c = 0; c < vector_size - 1; c++) {
                   gender_direction[c] += W[(word1_index_1 - 1) * (vector_size + 1) + c] - W[(word1_index_2 - 1) * (vector_size + 1) + c];
               }
           }
        }
        fprintf(stderr, "%d pairs will be used for gender direction\n", gender_direction_count);
        for (c = 0; c < vector_size - 1; c++) gender_direction[c] /= pair_count;

        //HASHREC *girl = hashsearch(vocab_hash,(char *)"girl");
        //fprintf(stderr, "word embeddings for \"girl\":\n");
        //int girl_index = girl->id;
        //for (c = 0; c < vector_size; c ++) fprintf(stderr, "%f ", W[(girl_index - 1) * (1 + vector_size) + c]);
        //fprintf(stderr, "\n");

        total_cost = 0;
        for (a = 0; a < num_threads - 1; a++) lines_per_thread[a] = num_lines / num_threads;
        lines_per_thread[a] = num_lines / num_threads + num_lines % num_threads;
        for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, glove_thread, (void *)a);
        for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
        for (a = 0; a < num_threads; a++) total_cost += cost[a];
        fprintf(stderr,"iter: %03d, cost: %lf\n", b+1, total_cost/num_lines);
    }
    return save_params();
}

int find_arg(char *str, int argc, char **argv) {
    int i;
    for (i = 1; i < argc; i++) {
        if(!scmp(str, argv[i])) {
            if (i == argc - 1) {
                printf("No argument given for %s\n", str);
                exit(1);
            }
            return i;
        }
    }
    return -1;
}

int main(int argc, char **argv) {


    int i;
    FILE *fid;
    vocab_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    input_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    male_word_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    female_word_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    vocab_hash_file= malloc(sizeof(char) * MAX_STRING_LENGTH);
    save_W_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    save_gradsq_file = malloc(sizeof(char) * MAX_STRING_LENGTH);

    if (argc == 1) {
        printf("GloVe: Global Vectors for Word Representation, v0.2\n");
        printf("Author: Jeffrey Pennington (jpennin@stanford.edu)\n\n");
        printf("Usage options:\n");
        printf("\t-verbose <int>\n");
        printf("\t\tSet verbosity: 0, 1, or 2 (default)\n");
        printf("\t-vector-size <int>\n");
        printf("\t\tDimension of word vector representations (excluding bias term); default 50\n");
        printf("\t-threads <int>\n");
        printf("\t\tNumber of threads; default 8\n");
        printf("\t-iter <int>\n");
        printf("\t\tNumber of training iterations; default 25\n");
        printf("\t-eta <float>\n");
        printf("\t\tInitial learning rate; default 0.05\n");
        printf("\t-alpha <float>\n");
        printf("\t\tParameter in exponent of weighting function; default 0.75\n");
        printf("\t-x-max <float>\n");
        printf("\t\tParameter specifying cutoff in weighting function; default 100.0\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave output in binary format (0: text, 1: binary, 2: both); default 0\n");
        printf("\t-model <int>\n");
        printf("\t\tModel for word vector output (for text output only); default 2\n");
        printf("\t\t   0: output all data, for both word and context word vectors, including bias terms\n");
        printf("\t\t   1: output word vectors, excluding bias terms\n");
        printf("\t\t   2: output word vectors + context word vectors, excluding bias terms\n");
        printf("\t-input-file <file>\n");
        printf("\t\tBinary input file of shuffled cooccurrence data (produced by 'cooccur' and 'shuffle'); default cooccurrence.shuf.bin\n");
        printf("\t-vocab-file <file>\n");
        printf("\t\tFile containing vocabulary (truncated unigram counts, produced by 'vocab_count'); default vocab.txt\n");
        printf("\t-save-file <file>\n");
        printf("\t\tFilename, excluding extension, for word vector output; default vectors\n");
        printf("\t-gradsq-file <file>\n");
        printf("\t\tFilename, excluding extension, for squared gradient output; default gradsq\n");
        printf("\t-save-gradsq <int>\n");
        printf("\t\tSave accumulated squared gradients; default 0 (off); ignored if gradsq-file is specified\n");
        printf("\nExample usage:\n");
        printf("./glove -input-file cooccurrence.shuf.bin -vocab-file vocab.txt -save-file vectors -gradsq-file gradsq -verbose 2 -vector-size 100 -threads 16 -alpha 0.75 -x-max 100.0 -eta 0.05 -binary 2 -model 2\n\n");
        return 0;
    }


    if ((i = find_arg((char *)"-verbose", argc, argv)) > 0) verbose = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-vector-size", argc, argv)) > 0) vector_size = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-iter", argc, argv)) > 0) num_iter = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    cost = malloc(sizeof(real) * num_threads);
    if ((i = find_arg((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = find_arg((char *)"-x-max", argc, argv)) > 0) x_max = atof(argv[i + 1]);
    if ((i = find_arg((char *)"-eta", argc, argv)) > 0) eta = atof(argv[i + 1]);
    if ((i = find_arg((char *)"-binary", argc, argv)) > 0) use_binary = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-model", argc, argv)) > 0) model = atoi(argv[i + 1]);
    if(model != 0 && model != 1) model = 2;
    if ((i = find_arg((char *)"-save-gradsq", argc, argv)) > 0) save_gradsq = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-vocab-file", argc, argv)) > 0) strcpy(vocab_file, argv[i + 1]);
    else strcpy(vocab_file, (char *)"vocab.txt");
    if ((i = find_arg((char *)"-save-file", argc, argv)) > 0) strcpy(save_W_file, argv[i + 1]);
    else strcpy(save_W_file, (char *)"vectors");
    if ((i = find_arg((char *)"-gradsq-file", argc, argv)) > 0) {
        strcpy(save_gradsq_file, argv[i + 1]);
        save_gradsq = 1;
    }
    else if(save_gradsq > 0) strcpy(save_gradsq_file, (char *)"gradsq");
    if ((i = find_arg((char *)"-input-file", argc, argv)) > 0) strcpy(input_file, argv[i + 1]);
    else strcpy(input_file, (char *)"cooccurrence.shuf.bin");

    if ((i = find_arg((char *)"-male-word-file", argc, argv)) > 0) strcpy(male_word_file, argv[i + 1]);
    else strcpy(male_word_file, (char *)"../wordlist/male_word_file.txt");
    if ((i = find_arg((char *)"-female-word-file", argc, argv)) > 0) strcpy(female_word_file, argv[i + 1]);
    else strcpy(female_word_file, (char *)"../wordlist/female_word_file.txt");

    if ((i = find_arg((char *)"-vocab-hash-file", argc, argv)) > 0) strcpy(vocab_hash_file, argv[i + 1]);
    else strcpy(vocab_hash_file, (char *)"hashdump.txt");

    vocab_size = 0;
    fid = fopen(vocab_file, "r");
    if(fid == NULL) {fprintf(stderr, "Unable to open vocab file %s.\n",vocab_file); return 1;}
    while ((i = getc(fid)) != EOF) if (i == '\n') vocab_size++; // Count number of entries in vocab_file
    fclose(fid);

    return train_glove();
}
