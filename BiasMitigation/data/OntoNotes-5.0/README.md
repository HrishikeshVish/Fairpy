# OntoNotes-5.0-NER-BIO

This is a CoNLL-2003 formatted version with BIO tagging scheme of the OntoNotes 5.0 release for NER. This formatted version is based on the instructions [here](http://cemantix.org/data/ontonotes.html) and a new script created in this repo. 

Simply put, the paper named *[Towards Robust Linguistic Analysis using OntoNotes](http://www.aclweb.org/anthology/W13-3516)* (Yuchen Zhang, Zhi Zhong, CoNLL 2013), proposed a train-dev-split for the OntoNotes 5.0 data, and provided scripts for converting it to CoNLL 2012 format. However, the results are not in BIO tagging scheme and cannot be directly used in most sequence tagging architectures, such as BLSTM-CRFs. This repo simplifies your pre-processing by directly generated the BIO format and you can use them in your experiments.



#### Step 1: Obtaining the official OntoNotes 5.0 release 

You can download the data at [https://catalog.ldc.upenn.edu/LDC2013T19](https://catalog.ldc.upenn.edu/LDC2013T19), and unpack the data.

`$ tar zxvf LDC2013T19.tgz`

#### Step 2: Running the script to recover words

The orginal repo of the authors only contains the  *_skel files, which mask the orginal words. Thus, you have to run the script to recover them from the downloaded OntoNotes 5.0 data. The script is based on python 2. If you are using conda, you can create a virtual environment for running python 2 quickly:
```
$ conda create --name py27 python=2.7.10
$ source activate py27
```

Make sure the command `python` in your terminal refers to a version of python 2.x. (You can check it by `$ which python`.)

Then, you can run: 
```
./conll-formatted-ontonotes-5.0/scripts/skeleton2conll.sh -D ../ontonotes-release-5.0/data/files/data ./conll-formatted-ontonotes-5.0
```
It assumes that you put the downloaded ontonotes data outside this repo but at the same level. This step can take some time (around 5 mins on my macbook pro). Now, you should have *_conll files corresponding to the original *_skel files.

You can deactive the envrionment now (`$ source deactivate`).

#### Step 3: Combine the data and convert tags within BIO.

Run `python3 agg.py` to obtain `onto.train.ner`, `onto.development.ner`, and `onto.test.ner`. （I only keep the first 50 lines as the examples here.)
