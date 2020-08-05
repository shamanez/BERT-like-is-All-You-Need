# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging

import numpy as np

import torch

from fairseq.data import (
    RawAudioTextDataset,
    RawAudioTextUnsupDataset,
    RoundRobinZipDatasets
)



from fairseq.tasks import FairseqTask, register_task
from collections import OrderedDict

logger = logging.getLogger(__name__)


@register_task('emotion_semisup_prediction')
class EmotionSemisupPredictionTask(FairseqTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """



    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('--data', metavar='FILE',
                            help='file prefix for Text data')

        parser.add_argument('--data-raw', metavar='FILE',
                            help='file prefix for Text data')

 
        parser.add_argument('--sample-rate', default=16000, type=int,
                            help='target sample rate. audio files will be up/down sampled to this rate')

        parser.add_argument('--max-sample-size', default=None, type=int,
                            help='max sample size to crop to for batching. default = min sample length')
        parser.add_argument('--min-sample-size', default=None, type=int,
                            help='min sample size to crop to for batching. default = same as --max-sample-size')

        parser.add_argument('--num-classes', type=int, default=-1,
                            help='number of classes')
        parser.add_argument('--init-token', type=int, default=None,
                            help='add token at the beginning of each batch item')
        parser.add_argument('--separator-token', type=int, default=None,
                            help='add separator token between inputs')
        parser.add_argument('--regression-target', action='store_true', default=False)
        parser.add_argument('--no-shuffle', action='store_true', default=False)
        parser.add_argument('--truncate-sequence', action='store_true', default=False,
                            help='Truncate sequence to max_sequence_length')

        ### cutomizing acording to the dataset
        parser.add_argument('--regression-target-mos', action='store_true', default=False)
        parser.add_argument('--binary-target-iemocap', action='store_true', default=False)
        parser.add_argument('--softmax-target-meld', action='store_true', default=False)
        parser.add_argument('--softmax-target-binary-meld', action='store_true', default=False)

        parser.add_argument('--eval-metric', action='store_true', default=False)  # This is to get the paper evaluations for each dataset
        parser.add_argument('--save-pred', action='store_true', default=False)
                            

    def __init__(self, args):
        super().__init__(args)

        #self.data_loaders=['sup', 'unsup'] #how many datasets we need to run
        self.data_loaders=['sup']  # two datasets right now


    @classmethod
    def setup_task(cls, args, **kwargs): # to_d0 :Might need to modify this also to merge audio and video
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        return cls(args)  #Initalization of an instance of our class

    def load_dataset(self, split, combine=False, **kwargs): 


     
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """   


        if split=='train':  #in the training phase we use both datasets

         

            self.datasets[split] = RoundRobinZipDatasets(
                OrderedDict([
                    (dataset_type, RawAudioTextDataset(base_path=self.args.data,
                                                    data_args=self.args, 
                                                    data_split=split, 
                                                    sample_rate=self.args.sample_rate,
                                                    max_sample_size=self.args.max_sample_size,
                                                    min_sample_size=self.args.min_sample_size))



                    if dataset_type=='sup' else (dataset_type, RawAudioTextDataset(base_path=self.args.data,
                                                    data_args=self.args, 
                                                    data_split=split, 
                                                    sample_rate=self.args.sample_rate,
                                                    max_sample_size=self.args.max_sample_size,
                                                    min_sample_size=self.args.min_sample_size))



                    # if dataset_type=='sup' else (dataset_type, RawAudioTextUnsupDataset(base_path=self.args.data,   
                    #                                 data_args=self.args, 
                    #                                 data_split=split, 
                    #                                 dataset_type=dataset_type,
                    #                                 sample_rate=self.args.sample_rate,
                    #                                 max_sample_size=self.args.max_sample_size,
                    #                                 min_sample_size=self.args.min_sample_size))



                                                    
                    for dataset_type in self.data_loaders
                ]))

    
        else:  #In the validation or testing phase we do not need any dataset
            # self.datasets[split] = RawAudioTextDataset(base_path=self.args.data,
            #                                        data_args=self.args, 
            #                                        data_split=split, 
            #                                        sample_rate=self.args.sample_rate,
            #                                        max_sample_size=self.args.max_sample_size,
            #                                        min_sample_size=self.args.min_sample_size)

            

            self.datasets[split] = RoundRobinZipDatasets(
                OrderedDict([
                    (dataset_type, RawAudioTextDataset(base_path=self.args.data,
                                                    data_args=self.args, 
                                                    data_split=split, 
                                                    sample_rate=self.args.sample_rate,
                                                    max_sample_size=self.args.max_sample_size,
                                                    min_sample_size=self.args.min_sample_size))

                    if dataset_type=='sup' else (dataset_type, RawAudioTextDataset(base_path=self.args.data,
                                                    data_args=self.args, 
                                                    data_split=split,
                                                    sample_rate=self.args.sample_rate,
                                                    max_sample_size=self.args.max_sample_size,
                                                    min_sample_size=self.args.min_sample_size))
                                                    
                    for dataset_type in self.data_loaders
                ]))



    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)

        
        loss, sample_size, logging_output = criterion(model, sample,update_num)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output


    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample,0)
        return loss, sample_size, logging_output

    # def train_step(self, sample, model, criterion, optimizer,update_num ,ignore_grad=False): #no need a seperate valid step

      
    #     model.train()
    #     from collections import defaultdict
    #     agg_loss, agg_sample_size, agg_logging_output = 0., 0., defaultdict(float)



    #     for data_set_type in self.data_loaders:

    #         print(data_set_type,len(sample[data_set_type]))

    #         continue

          
    #         if sample[data_set_type] is None or len(sample[data_set_type]) == 0:
    #             continue
    #         #loss, sample_size, logging_output = criterion(model.models[lang_pair], sample[lang_pair])
    #         loss, sample_size, logging_output = criterion(model, sample[data_set_type])
         

       
    #         if ignore_grad:
    #             loss *= 0
    #         optimizer.backward(loss)
    #         agg_loss += loss.detach().item()
    #         # TODO make summing of the sample sizes configurable
    #         agg_sample_size += sample_size
    #         for k in logging_output:
    #             agg_logging_output[k] += logging_output[k]
    #             #agg_logging_output[f"{data_set_type}:{k}"] += logging_output[k]
    #     return agg_loss, agg_sample_size, agg_logging_output

      

    def build_model(self, args):
        
        from fairseq import models
      
        model = models.build_model(args, self) #/home/gsir059/Documents/PhD/MulFie/fairseq/fairseq/models/__init__.py'
        print("Model Initialization Done")

        model.register_classification_head(
            getattr(args, 'classification_head_name', 'emotion_classification_head'),
            num_classes=self.args.num_classes,
        )
    
        # model.register_classification_head(
        #     'emotion_classification_head',
        #     num_classes=self.args.num_classes,
        # )
      
        return model

    # def max_positions(self):
    #     return self.args.max_positions

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None