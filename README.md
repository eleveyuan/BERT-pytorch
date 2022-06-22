# BERT-pytorch
PyTorch implementation of BERT in "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (https://arxiv.org/abs/1810.04805)

## Requirements
* python3
* pytorch 

All dependencies can be installed via:
``` shell
pip install -r requirements.txt
```

## Preprocess
First things first, you need to prepare your data in an appropriate format. 
Your corpus is assumed to follow the below constraints.

- Each line is a *document*.
- A *document* consists of *sentences*, seperated by vertical bar (|).
- A *sentence* is assumed to be already tokenized. Tokens are seperated by space.
- A *sentence* has no more than 256 tokens.
- A *document* has at least 2 sentences. 
- You have two distinct data files, one for train data and the other for val data.

This repo comes with example data for pretraining in data/example directory.
Here is the content of data/example/train.txt file.

```
One, two, three, four, five,|Once I caught a fish alive,|Six, seven, eight, nine, ten,|Then I let go again.
I’m a little teapot|Short and stout|Here is my handle|Here is my spout.
Jack and Jill went up the hill|To fetch a pail of water.|Jack fell down and broke his crown,|And Jill came tumbling after.  
```

Also, this repo includes SST-2 data in data/SST-2 directory for sentiment classification.


CAUTION: Whole process will take your lots of time. 

### Step by Step

``` shell
python main.py extract-wiki --wiki_raw_path data/wiki-example/enwiki-latest-pages-articles.xml.bz2

python main.py detect-sentences --raw_documents_path data/wiki-example/raw_documents.txt 

python main.py split-sentences --sentences_detected_path data/wiki-example/sentences_detected.txt

python main.py train-tokenizer --spm_input_path data/wiki-example/spm_input.txt

python main.py prepare-documents --sentences_detected_path data/wiki-example/sentences_detected.txt

python main.py split-train-val --prepared_documents_path data/wiki-example/prepared_documents.txt

python main.py build-dictionary --train_path data/wiki-example/train.txt
```
when you are trianing tokenizer on the forth step, in case memory stackoverflow or other uncontrolled thing, you'd better set parameters `--input_sentence_size=5000000` and `--shuffle_input_sentence=false`. 

### All-in-One

``` shell
python main.py preprocess-all --data_dir data/wiki-example/
```

## Pre-train the model
```
python main.py pretrain --train_data data/example/train.txt --val_data data/example/val.txt --checkpoint_output model.pth
```
This step trains BERT model with unsupervised objective. Also this step does:
- logs the training procedure for every epoch
- outputs model checkpoint periodically
- reports the best checkpoint based on validation metric

## Fine-tune the model
You can fine-tune pretrained BERT model with downstream task.
For example, you can fine-tune your model with SST-2 sentiment classification task. 
```
python main.py finetune --pretrained_checkpoint model.pth --train_data data/SST-2/train.tsv --val_data data/SST-2/dev.tsv
```
This command also logs the procedure, outputs checkpoint, and reports the best checkpoint.
