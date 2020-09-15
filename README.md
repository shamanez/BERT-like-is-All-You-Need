# [Jointly Fine-Tuning “BERT-like” Self Supervised Models to Improve Multimodal Speech Emotion Recognition](https://arxiv.org/abs/2008.06682)

![Model Overviw](https://github.com/shamanez/BERT-like-is-All-You-Need/blob/master/pipeline.jpg)

This repositary consist the pytorch code for Multimodal Emotion Recogntion with pretreined Roberta and Speech-BERT.


# Basic strucutre of the code

## Inspiration from fairseq

1. This code strcuture is built on top of Faiseq interface
2. Fairseq is an open source project by FacebookAI team that combined different SOTA architectures for sequencial data processing
3. This also consist of SOTA optimizing mechanisms such as ealry stopage, warup learnign rates, learning rate shedulers
4. We are trying to develop our own architecture in compatible with fairseq interface. 
5. For more understanding please read the [paper](https://arxiv.org/abs/1904.01038) published about Fairseq interaface.

## Merging of our own architecture with Fairseq interface

1. This can be bit tricky in the beggining. First  it is important to udnestand that Fairseq has built in a way that all architectures can be access through the terminal commands (args).

2. Since our architecture has lot of properties in tranformer architecture, we followed the [a tutorial](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.custom_classification.md) that describe to use Roberta for the custom classification task.

3. We build over archtiecture by inserting new stuff to following directories in Fairseq interfeace.
   - fairseq/data
   - fairseq/models
   - fairseq/modules
   - fairseq/tasks
   - fairseq/criterions


# Main scripts of the code

## Our main scripts are categorized in to for parts

1. Custom dataloader for load raw audio, faceframes and text is in the **fairseq/data/raw_audio_text_dataset.py**

2. The task of the emotion prediction similar to other tasks such as translation is in the **fairseq/tasks/emotion_prediction.py**

3. The custom architecture of our model similar to roberta,wav2vec is in the **fairseq/models/mulT_emo.py**

4. The cross-attention was implemted by modifying the self attentional scripts in original fairseq repositary. They can be found in **fairseq/modules/transformer_multi_encoder.py** and  **fairseq/modules/transformer_layer.py**

5. Finally the cutom loss function and ebaluation scripts can be found it **fairseq/criterions/emotion_prediction_cri.py**



# Prerequest models 

### Please use following links to downlaod the pretrained SSL models and save them in a seperate folder named pretrained_ssl.

1. For speech fetures - [VQ-wav2vec](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec) 
2. For sentence (text) features - [Roberta](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md)


# Preprocessing data.

### We tokenized both speech and text data and then feed in to the algorithm training.

1. For text data, we first tokenized it with Roberta tokenizer and save each example in to seperate text files.
2. To preprocess speech data please refer the script given in [convert_aud_to_token.py](https://github.com/shamanez/BERT-like-is-All-You-Need/tree/master/SPEECH-BERT-TOKENIZATION).
3. The preprocessed datasets and their labels can be found in the [this google drive](https://drive.google.com/drive/folders/1oiAiY0QgIpP3Wb9bPfC4hYdGrJ1MmDAP?usp=sharing).



# Terminal Commands 

We followed the Fairseq terminal commands to train and validate our models.

## Useful commands 

1. --data - folder that contains filenames, sizes and labels of your raw data (please refer to the T_data folder). 
2. --data-raw - Path of your raw data folder that contains tokenized speech and text.
3. --binary-target-iemocap - train the model with Iemocap data for binary accuracy.
4. --regression-target-mos - train the model with CMU-MOSEI/CMU-MOSI data for sentiment score.
5. For dataset specific traing commands please refer to [emotion_prediction.py](https://github.com/shamanez/BERT-like-is-All-You-Need/blob/master/fairseq/tasks/emotion_prediction.py).

## Training Command

CUDA_VISIBLE_DEVICES=8,7  python train.py --data ./T_data/iemocap --restore-file None  --task emotion_prediction --reset-optimizer --reset-dataloader --reset-meters --init-token 0 --separator-token 2 --arch robertaEMO --criterion  emotion_prediction_cri  --num-classes 8  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 --clip-norm 0.0 --lr-scheduler polynomial_decay --lr 1e-05 --total-num-update 2760  --warmup-updates 165  --max-epoch 10 --best-checkpoint-metric loss  --encoder-attention-heads 2 --batch-size 1 --encoder-layers-cross 1   --no-epoch-checkpoints --update-freq 8 --find-unused-parameters --ddp-backend=no_c10d --binary-target-iemocap    --a-only --t-only  --pooler-dropout 0.1  --log-interval 1  --data-raw ./iemocap_data/  

## Validation Command


CUDA_VISIBLE_DEVICES=1 python validate.py  --data ./T_data/iemocap   --path './checkpoints/checkpoint_best.pt' --task emotion_prediction --valid-subset test --batch-size 4
