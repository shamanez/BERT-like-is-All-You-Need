# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

import logging
import numpy as np

from fairseq import utils
from fairseq.models import (
    FairseqDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
    TransformerSentenceEncoder,
    TransformerMultiEncoder
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
#from fairseq.models.roberta import RobertaModel
from fairseq import checkpoint_utils
import pdb
from fairseq.models.roberta import RobertaModel


logger = logging.getLogger(__name__)
#from .hub_interface import RobertaHubInterface


@register_model('semiemo')
class SemiemoModel(FairseqLanguageModel):


    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

        ############################## Adding the pretrained SSL models to extract features###############


        if self.args.a_only or self.args.all_in:
           
            self.roberta_vqwav2vec = RobertaModel.from_pretrained('/hpc/gsir059/INTERSPEECH/MOSI-SEMI/trained_ssl/wav2vec/vq-wav2vec-Kmeans-Roberta',checkpoint_file='bert_kmeans.pt')

            if self.args.frozen_ssl:
                for param in self.roberta_vqwav2vec.parameters():
                    param.requires_grad = False


        if self.args.t_only or self.args.all_in:
            roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')

        
            ########################### Freezing pretrained SSL paramtere###################################
            self.model_text2vec=roberta
            if self.args.frozen_ssl:

                for param in self.model_text2vec.parameters():
                    param.requires_grad = False



       

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-layers', type=int, metavar='L',
                            help='num encoder layers in plain trans')
        parser.add_argument('--encoder-layers-cross', type=int, metavar='L',
                            help='num encoder layers in cross modal trans')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='H',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-dim-concat', type=int, metavar='H',
                            help='encoder embedding dimension concatenated')
        parser.add_argument('--encoder-embed-dim-a', type=int, metavar='H',
                            help='encoder embedding dimension cross modal audio')
        parser.add_argument('--encoder-embed-dim-v', type=int, metavar='H',
                            help='encoder embedding dimension cross modal audio')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='F',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='A',
                            help='num encoder attention heads')
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--pooler-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use for pooler layer')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN')
        parser.add_argument('--pooler-dropout', type=float, metavar='D',
                            help='dropout probability in the masked_lm pooler layers')
        parser.add_argument('--max-positions-t', type=int,
                            help='number of positional embeddings to learn')
        parser.add_argument('--load-checkpoint-heads', action='store_true',
                            help='(re-)register and load heads when loading checkpoints')

        parser.add_argument('--max-positions-v', type=int,
                            help='number of positional embeddings to learn in video stream')
        parser.add_argument('--max-positions-a', type=int,
                            help='number of positional embeddings to learn in audio stream')


        parser.add_argument('--t-only', action='store_true', default=False,
                            help='do you need only text')

        parser.add_argument('--v-only', action='store_true', default=False,
                                    help='do you need only video')

        parser.add_argument('--a-only', action='store_true', default=False,
                                    help='do you need only audio')


        parser.add_argument('--all-in', action='store_true', default=False,
                                    help='do you need all the embeddings')

        parser.add_argument('--stack-up', action='store_true', default=False,
                                    help='do you need to add an architecture on top of SSL layers')

        parser.add_argument('--frozen-ssl', action='store_true', default=False,
                                    help='do you need to keep the ssl frozen')


    
    def set_num_updates(self, num_updates):

        self.curr_step = num_updates


    
    @classmethod
    def build_model(cls, args, task):


        """Build a new model instance."""
        # make sure all arguments are present
        #base_architecture(args) #Supplying the relevence arguments to create the baseline transformer architecture

       

        encoder = SemiemoEncoder(args, task.source_dictionary)
     
        return cls(args, encoder)

    def forward(self, src, features_only=False, return_all_hiddens=False, classification_head_name=None, **kwargs):

        final_loss_input = {
            'sup_logits': None, #we can take sup targets as usual 
            'sup_targets': None,
            'ori_uda_logits': None,
            'aug_uda_logits': None,
            'simCLR_soft_logits': None,
            'simCLR_soft_labels': None,
            'mixup_logits':None,
            'mixup_targets':None,
            'sup_split':None,

        }



        sup_src=src['sup']
        #unsup_src=src['unsup']
        unsup_src=None

       
        if sup_src is None or len(sup_src) == 0:
            sup_src_tokens=None
        
        else:
            sup_src_tokens=sup_src['net_input']
            sup_src_target=sup_src['target']
            sup_src_split=sup_src['split']
            final_loss_input.update(sup_split=sup_src_split)




        if unsup_src is None or len(unsup_src) == 0:
            unsup_src_tokens=None
        
        else:
            unsup_src_tokens=unsup_src['net_input']

        

        #unsup_src_tokens=None

        if sup_src_tokens is not None:

            #batch_size_sup=sup_src_target.shape[0]

            aug=False # because we are not augmenting anything

            if classification_head_name is not None:
                features_only = True
            

            data_dict_sup={} #Create a dictionary with features
            data_dict_sup['raw_data']=sup_src_tokens

            if self.args.t_only or self.args.all_in:
                tokens_text_sup=sup_src_tokens['text']

    
                #Text SSL feature extraction  # [2, 100, 1024] B X T X D
                roberta_feature_sup=self.model_text2vec.extract_features(tokens_text_sup)
                data_dict_sup['Text']=roberta_feature_sup
            
                    

            if self.args.a_only or self.args.all_in:

                tokens_audio_sup=sup_src_tokens['audio']

                roberta_vqwav2vec_feature_sup=self.roberta_vqwav2vec.extract_features(tokens_audio_sup)
        
                data_dict_sup['Audio']=roberta_vqwav2vec_feature_sup
        

                #Audio SSL feature extraction [2, 512, 310] B X D X T


            #This will output the main models whole features as well as token features
            x, extr_sup = self.decoder(data_dict_sup,features_only, return_all_hiddens,aug, **kwargs) #here the decoder means the encoder (to have the interface fixed)
            
            if classification_head_name is not None:
            
                x_sup,_ = self.classification_heads[classification_head_name](extr_sup,Final_rep=False) #Here we send the extracs ("First token")

         
            final_loss_input.update(sup_logits=x_sup,sup_targets=sup_src_target)

    
            

        # if unsup_src_tokens is not None:

        #     aug=True
        
        #     data_dict_unsup={} #Create a dictionary with features
        #     data_dict_unsup['raw_data']=unsup_src_tokens

        #     if self.args.t_only or self.args.all_in:
        #         tokens_text_unsup=unsup_src_tokens['text']
            

        #         #Text SSL feature extraction  # [2, 100, 1024] B X T X D
        #         roberta_feature_unsup=self.model_text2vec.extract_features(tokens_text_unsup)
        #         data_dict_unsup['Text']=roberta_feature_unsup
        

        #     if self.args.a_only or self.args.all_in:

        #         tokens_audio_unsup=unsup_src_tokens['audio']

        #         roberta_vqwav2vec_feature_unsup=self.roberta_vqwav2vec.extract_features(tokens_audio_unsup)
    
        #         data_dict_unsup['Audio']=roberta_vqwav2vec_feature_unsup
        

        #         #Audio SSL feature extraction [2, 512, 310] B X D X T

        #     if classification_head_name is not None:
        #         features_only = True

        #     # ori_u_score, aug_u_score= self.UDA_unsup(data_dict_unsup,classification_head_name,return_all_hiddens)

        #     # final_loss_input.update(ori_uda_logits=ori_u_score,aug_uda_logits=aug_u_score)

        
            
        # if (unsup_src_tokens is not None) and (sup_src_tokens is not None):

         
        #     mixup_logits,mixed_target=self.rep_mix_up_BA(data_dict_sup, data_dict_unsup,sup_src_target,classification_head_name,return_all_hiddens)
        #     final_loss_input.update(mixup_logits=mixup_logits,mixup_targets=mixed_target)

        #     #print("no mixmatch")

        #     #use cross entropy
        #     # contrastive_logits,contrastive_labels= self.SimCLR_unsup(data_dict_sup, data_dict_unsup,sup_src_target,classification_head_name,return_all_hiddens)
        #     # final_loss_input.update(simCLR_soft_logits=contrastive_logits,simCLR_soft_labels=contrastive_labels)

            
            
            



        return final_loss_input

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""

        
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                print(
                    'WARNING: re-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )

    
        self.classification_heads[name] = SemiemoClassificationHead(
            self.args.encoder_embed_dim_concat,#self.args.encoder_embed_dim,
            #inner_dim or self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim_concat,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
            self.args
        )



    def rep_mix_up(self, sup_sample, unsup_sample, targets_x, classification_head_name, return_all_hiddens, **kwargs):



            targets_x=torch.nn.functional.one_hot(targets_x.long(), 4).float()

            #################################################################
            ##only for iemocap
            #targets_x= targets_x.view(-1)
            #targets_x=torch.nn.functional.one_hot(targets_x.long(), 2).float()

            ###################################################################

            Temparature=0.5

            if classification_head_name is not None:
                features_only = True

            
            if self.args.stack_up:

        
                with torch.no_grad(): #get the average over the predictions for two augmented versions
                    # compute guessed labels of unlabel samples
                    features_only=True
                    _,rep_u =self.decoder(unsup_sample,features_only, return_all_hiddens,aug=True, **kwargs) #randomly we get two aug
                    _,rep_u2 =self.decoder(unsup_sample,features_only, return_all_hiddens,aug=True, **kwargs)

                    outputs_u1,_ = self.classification_heads[classification_head_name](rep_u,Final_rep=False) #from the usual setting
                    outputs_u2,_ = self.classification_heads[classification_head_name](rep_u2,Final_rep=False) #change the text  [cls] position littlebit

                    ############################################################
                    # outputs_u1=outputs_u1.view(-1, 2) #only for iemocap
                    # outputs_u2=outputs_u2.view(-1, 2)
                    ##################################################################################

                    average_u_score_ori = (torch.softmax(outputs_u1, dim=1)+torch.softmax(outputs_u2, dim=1)) / 2
                    pt = average_u_score_ori**(1/Temparature)
                    targets_u=pt / pt.sum(dim=1, keepdim=True)
                    targets_u = targets_u.detach()

                _,rep_unsup =self.decoder(unsup_sample,features_only, return_all_hiddens,aug=True, **kwargs)
                _,final_unsup = self.classification_heads[classification_head_name](rep_unsup,Final_rep=False)

                _,rep_unsup2 =self.decoder(unsup_sample,features_only, return_all_hiddens,aug=True, **kwargs)
                _,final_unsup2 = self.classification_heads[classification_head_name](rep_unsup2,Final_rep=False)

                _,rep_sup =self.decoder(sup_sample,features_only, return_all_hiddens,aug=False, **kwargs) #if we need we can put the  augmentation true
                _,final_sup = self.classification_heads[classification_head_name](rep_sup,Final_rep=False)



                all_inputs = torch.cat([final_sup,final_unsup,final_unsup2], dim=0)  #make a mixed batch
                all_targets = torch.cat([targets_x, targets_u,targets_u], dim=0) #all the examples are together
   

                # ############################################################################################
                # all_targets=all_targets.view(all_inputs.shape[0],-1) #only for binary emocap thing

            alpha=0.75
            Temparature=0.5

            l = np.random.beta(alpha, alpha)
            l = max(l, 1-l)      
            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]  
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b  #mixing the inputs with 
            mixed_target = l * target_a + (1 - l) * target_b
            mixup_logits,_=self.classification_heads[classification_head_name](mixed_input,Final_rep=True)

     
            return mixup_logits,mixed_target

    def rep_mix_up_BA(self, sup_sample, unsup_sample, targets_x, classification_head_name, return_all_hiddens, **kwargs):



            targets_x=torch.nn.functional.one_hot(targets_x.long(), 4).float()
            ################################################################
            #only for iemocap
            targets_x= targets_x.view(-1)
            targets_x=torch.nn.functional.one_hot(targets_x.long(), 2).float()
            ##################################################################

     

            Temparature=0.5

            if classification_head_name is not None:
                features_only = True

            
            if self.args.stack_up:

                with torch.no_grad(): #get the average over the predictions for two augmented versions
                    # compute guessed labels of unlabel samples
                    features_only=True
                    _,rep_u =self.decoder(unsup_sample,features_only, return_all_hiddens,aug=True, **kwargs) #randomly we get two aug
                    _,rep_u2 =self.decoder(unsup_sample,features_only, return_all_hiddens,aug=True, **kwargs)

                    outputs_u1,_ = self.classification_heads[classification_head_name](rep_u,Final_rep=False) #from the usual setting
                    outputs_u2,_ = self.classification_heads[classification_head_name](rep_u2,Final_rep=False) #change the text  [cls] position littlebit

                    ###########################################################
                    outputs_u1=outputs_u1.view(-1, 2) #only for iemocap
                    outputs_u2=outputs_u2.view(-1, 2)
                    #################################################################################


                    average_u_score_ori = (torch.softmax(outputs_u1, dim=1)+torch.softmax(outputs_u2, dim=1)) / 2
                    pt = average_u_score_ori**(1/Temparature)
                    targets_u=pt / pt.sum(dim=1, keepdim=True)
                    targets_u = targets_u.detach()

                _,rep_unsup =self.decoder(unsup_sample,features_only, return_all_hiddens,aug=True, **kwargs)
                _,final_unsup = self.classification_heads[classification_head_name](rep_unsup,Final_rep=False)

                _,rep_unsup2 =self.decoder(unsup_sample,features_only, return_all_hiddens,aug=True, **kwargs)
                _,final_unsup2 = self.classification_heads[classification_head_name](rep_unsup2,Final_rep=False)

                _,rep_sup =self.decoder(sup_sample,features_only, return_all_hiddens,aug=False, **kwargs) #if we need we can put the  augmentation true
                _,final_sup = self.classification_heads[classification_head_name](rep_sup,Final_rep=False)



                all_inputs = torch.cat([final_sup,final_unsup,final_unsup2], dim=0)  #make a mixed batch
                all_targets = torch.cat([targets_x, targets_u,targets_u], dim=0) #all the examples are together

                # ############################################################################################
                all_targets=all_targets.view(all_inputs.shape[0],-1) #only for binary emocap thing

   

            alpha=0.75
            Temparature=0.5

            l = np.random.beta(alpha, alpha)
            l = max(l, 1-l)      
            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]  
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b  #mixing the inputs with 
            mixed_target = l * target_a + (1 - l) * target_b
            mixup_logits,_=self.classification_heads[classification_head_name](mixed_input,Final_rep=True)

  
     
            return mixup_logits,mixed_target


    def rep_mix_up_no_stack(self, sup_sample, unsup_sample, targets_x, classification_head_name, return_all_hiddens, **kwargs):



            targets_x=torch.nn.functional.one_hot(targets_x.long(), 4).float()

            #################################################################
            ##only for iemocap
            #targets_x= targets_x.view(-1)
            #targets_x=torch.nn.functional.one_hot(targets_x.long(), 2).float()

            ###################################################################

            Temparature=0.5

            if classification_head_name is not None:
                features_only = True

            
            if self.args.stack_up:

        
                with torch.no_grad(): #get the average over the predictions for two augmented versions
                    # compute guessed labels of unlabel samples
                    features_only=True
                    _,rep_u =self.decoder(unsup_sample,features_only, return_all_hiddens,aug=False, **kwargs) #randomly we get two aug
                    _,rep_u2 =self.decoder(unsup_sample,features_only, return_all_hiddens,aug=False, **kwargs)

                    outputs_u1,_ = self.classification_heads[classification_head_name](rep_u,Final_rep=False) #from the usual setting
                    outputs_u2,_ = self.classification_heads[classification_head_name](rep_u2,Final_rep=False) #change the text  [cls] position littlebit

                    ############################################################
                    # outputs_u1=outputs_u1.view(-1, 2) #only for iemocap
                    # outputs_u2=outputs_u2.view(-1, 2)
                    ##################################################################################

                    average_u_score_ori = (torch.softmax(outputs_u1, dim=1)+torch.softmax(outputs_u2, dim=1)) / 2
                    pt = average_u_score_ori**(1/Temparature)
                    targets_u=pt / pt.sum(dim=1, keepdim=True)
                    #targets_u = targets_u.detach()

                _,rep_unsup =self.decoder(unsup_sample,features_only, return_all_hiddens,aug=True, **kwargs)
                _,final_unsup = self.classification_heads[classification_head_name](rep_unsup,Final_rep=False)

                _,rep_unsup2 =self.decoder(unsup_sample,features_only, return_all_hiddens,aug=True, **kwargs)
                _,final_unsup2 = self.classification_heads[classification_head_name](rep_unsup2,Final_rep=False)

                _,rep_sup =self.decoder(sup_sample,features_only, return_all_hiddens,aug=False, **kwargs) #if we need we can put the  augmentation true
                _,final_sup = self.classification_heads[classification_head_name](rep_sup,Final_rep=False)



                all_inputs = torch.cat([final_sup,final_unsup,final_unsup2], dim=0)  #make a mixed batch
                all_targets = torch.cat([targets_x, targets_u,targets_u], dim=0) #all the examples are together
   

                # ############################################################################################
                # all_targets=all_targets.view(all_inputs.shape[0],-1) #only for binary emocap thing

            alpha=0.75
            Temparature=0.5

            l = np.random.beta(alpha, alpha)
            l = max(l, 1-l)      
            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]  
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b  #mixing the inputs with 
            mixed_target = l * target_a + (1 - l) * target_b
            mixup_logits,_=self.classification_heads[classification_head_name](mixed_input,Final_rep=True)

            return mixup_logits,mixed_target



    def manifold_mix_up(self, sup_sample, unsup_sample, targets_x, classification_head_name, return_all_hiddens, **kwargs):



            targets_x=torch.nn.functional.one_hot(targets_x.long(), 4).float()
            targets_x_hat=torch.nn.functional.one_hot(unsup_sample['targets'].long(), 4).float()

            #################################################################
            ##only for iemocap
            #targets_x= targets_x.view(-1)
            #targets_x=torch.nn.functional.one_hot(targets_x.long(), 2).float()

            ###################################################################

            Temparature=0.5

            # if classification_head_name is not None:
            #     features_only = True

            
            # if self.args.stack_up:

        
            #     with torch.no_grad(): #get the average over the predictions for two augmented versions
            #         # compute guessed labels of unlabel samples
            #         features_only=True
            #         _,rep_u =self.decoder(unsup_sample,features_only, return_all_hiddens,aug=True, **kwargs) #randomly we get two aug
            #         _,rep_u2 =self.decoder(unsup_sample,features_only, return_all_hiddens,aug=True, **kwargs)

            #         outputs_u1,_ = self.classification_heads[classification_head_name](rep_u,Final_rep=False) #from the usual setting
            #         outputs_u2,_ = self.classification_heads[classification_head_name](rep_u2,Final_rep=False) #change the text  [cls] position littlebit

            #         ############################################################
            #         # outputs_u1=outputs_u1.view(-1, 2) #only for iemocap
            #         # outputs_u2=outputs_u2.view(-1, 2)
            #         ##################################################################################

            #         average_u_score_ori = (torch.softmax(outputs_u1, dim=1)+torch.softmax(outputs_u2, dim=1)) / 2
            #         pt = average_u_score_ori**(1/Temparature)
            #         targets_u=pt / pt.sum(dim=1, keepdim=True)
            #         targets_u = targets_u.detach()


            _,rep_unsup =self.decoder(unsup_sample,features_only, return_all_hiddens,aug=True, **kwargs)
            _,final_unsup = self.classification_heads[classification_head_name](rep_unsup,Final_rep=False)

            _,rep_unsup2 =self.decoder(unsup_sample,features_only, return_all_hiddens,aug=True, **kwargs)
            _,final_unsup2 = self.classification_heads[classification_head_name](rep_unsup2,Final_rep=False)

            _,rep_sup =self.decoder(sup_sample,features_only, return_all_hiddens,aug=False, **kwargs) #if we need we can put the  augmentation true
            _,final_sup = self.classification_heads[classification_head_name](rep_sup,Final_rep=False)


            all_inputs = torch.cat([final_sup,final_unsup,final_unsup2], dim=0)  #make a mixed batch
            all_targets = torch.cat([targets_x, targets_u,targets_u], dim=0) #all the examples are together
   

            alpha=0.75
            Temparature=0.5

            l = np.random.beta(alpha, alpha)
            l = max(l, 1-l)      
            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]  
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b  #mixing the inputs with 
            mixed_target = l * target_a + (1 - l) * target_b
            mixup_logits,_=self.classification_heads[classification_head_name](mixed_input,Final_rep=True)

     
            return mixup_logits,mixed_target




    def UDA_unsup(self, unsup_sample,classification_head_name,return_all_hiddens, **kwargs):

        if classification_head_name is not None:
            features_only = True


        with torch.no_grad(): #get the average over the predictions for two augmented versions
            # compute guessed labels of unlabel samples
            _,rep_u_ori =self.decoder(unsup_sample,features_only, return_all_hiddens,aug=False, **kwargs) #randomly we get two aug
            outputs_u1_ori,_ = self.classification_heads[classification_head_name](rep_u_ori,Final_rep=False) #from the usual setting



        _,rep_u =self.decoder(unsup_sample,features_only, return_all_hiddens,aug=True, **kwargs) #randomly we get two aug
        outputs_u1,_ = self.classification_heads[classification_head_name](rep_u,Final_rep=False) #from the usual setting

        ############################

        return outputs_u1,outputs_u1_ori

    def SimCLR_unsup(self, sup_sample, unsup_sample,targets_x,classification_head_name,return_all_hiddens, **kwargs):

        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask(targets_x.shape[0]*2).type(torch.bool)



        if classification_head_name is not None:
            features_only = True

        _,rep_u_dict =self.decoder(unsup_sample,features_only, return_all_hiddens,aug=True, **kwargs) #randomly we get two aug
        _,rep_u2_dict =self.decoder(unsup_sample,features_only, return_all_hiddens,aug=True, **kwargs)

        _,final_rep_u = self.classification_heads[classification_head_name](rep_u_dict,Final_rep=False)
        _,final_rep_u2=self.classification_heads[classification_head_name](rep_u2_dict,Final_rep=False)

        _,rep_sup_dict =self.decoder(sup_sample,features_only, return_all_hiddens,aug=True, **kwargs) #newly added
        _,rep_sup2_dict =self.decoder(sup_sample,features_only, return_all_hiddens,aug=True, **kwargs)#newly added

        _,final_rep_sup = self.classification_heads[classification_head_name](rep_sup_dict,Final_rep=False)#newly added
        _,final_rep_sup2=self.classification_heads[classification_head_name](rep_sup2_dict,Final_rep=False)#newly added

        
        final_rep=torch.cat([final_rep_sup,final_rep_u], dim=0) 
        final_rep2=torch.cat([final_rep_sup2,final_rep_u2], dim=0) 


     


        # normalize projection feature vectors
        # repis = F.normalize(final_rep_u, dim=1)  #first augmetation copy of the batch
        # repjs = F.normalize(final_rep_u2, dim=1) #second augmentation copy of the batch
        repis = F.normalize(final_rep, dim=1)  #first augmetation copy of the batch
        repjs = F.normalize(final_rep2, dim=1) #second augmentation copy of the batch
        representations = torch.cat([repis, repjs], dim=0)

    
        similarity_matrix = self._cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0)) #squeeze is really important

    

        batch_size=targets_x.shape[0]*2
        temperature=0.5


        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)


        logits = logits/temperature
        labels = torch.zeros(2 * batch_size).long() #every time it is the correct location 

   

        #use cross entropy loss as normal - reduction sum 
        #make sure that the batch size from this is twwise
   
        return logits,labels



    def _get_correlated_mask(self,batch_size): #for the SimCLR method
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask  #.to(self.device)




    @property
    def supported_targets(self):
        return {'self'}


    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''
        current_head_names = [] if not hasattr(self, 'classification_heads') else \
            self.classification_heads.keys()

     

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            #print(k)

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            #inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)
            #Because we do not have a dense layer
            inner_dim =state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(1)

    

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    print(
                        'WARNING: deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    
                    num_classes != self.classification_heads[head_name].out_proj.out_features
                    # or inner_dim != self.classification_heads[head_name].dense.out_features
                    or inner_dim != self.classification_heads[head_name].out_proj.in_features #because we do not have a dense layer
                    
                ):
                    print(
                        'WARNING: deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)

      
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    print('Overwriting', prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v








def upgrade_state_dict_named(self, state_dict, name):
    super().upgrade_state_dict_named(state_dict, name)

    prefix = name + '.' if name != '' else ''
    current_head_names = [] if not hasattr(self, 'classification_heads') else \
        self.classification_heads.keys()

    # Handle new classification heads present in the state dict.
    keys_to_delete = []
    for k in state_dict.keys():
        if not k.startswith(prefix + 'classification_heads.'):
            continue

        head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
        num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
        #inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)
        inner_dim =state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(1)

        if getattr(self.args, 'load_checkpoint_heads', False):
            if head_name not in current_head_names:
                self.register_classification_head(head_name, num_classes, inner_dim)
        else:
            if head_name not in current_head_names:
                logger.warning(
                    'deleting classification head ({}) from checkpoint '
                    'not present in current model: {}'.format(head_name, k)
                )
                keys_to_delete.append(k)
            elif (
                num_classes != self.classification_heads[head_name].out_proj.out_features
                or inner_dim != self.classification_heads[head_name].dense.out_features
            ):
                logger.warning(
                    'deleting classification head ({}) from checkpoint '
                    'with different dimensions than current model: {}'.format(head_name, k)
                )
                keys_to_delete.append(k)
    for k in keys_to_delete:
        del state_dict[k]

    # Copy any newly-added classification heads into the state dict
    # with their current weights.
    if hasattr(self, 'classification_heads'):
        cur_state = self.classification_heads.state_dict()
        for k, v in cur_state.items():
            if prefix + 'classification_heads.' + k not in state_dict:
                logger.info('Overwriting ' + prefix + 'classification_heads.' + k)
                state_dict[prefix + 'classification_heads.' + k] = v






class SemiemoClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout,args):
        super().__init__()

        #inner_dim needs to change

        #self.dense = nn.Linear(input_dim, inner_dim)
        #self.activation_fn = utils.get_activation_fn(activation_fn) #Activation function is Tanh
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)


        self.args=args

    def forward(self, features,Final_rep, **kwargs):


        if not Final_rep: #we do not get the final representation drectly

            if self.args.t_only & (not self.args.stack_up):
                Final=features['j_text']


            if self.args.a_only & (not self.args.stack_up) :
                Final=features['j_aud']

            
        

            
            if (self.args.all_in) or (self.args.a_only and self.args.t_only):


                if self.args.stack_up:

                    #exit("haha check the stack up section")

                    #exit("###############################################################")

                    T_A=features['t2a_r']
                    A_T=features['a2t_r']
                    Final=torch.cat((T_A,A_T),dim=1)

                else:
                    T_A=features['j_text']
                    A_T=features['j_aud']
                    Final=torch.cat((T_A,A_T),dim=1)


        else: #we get the final represetation directly mix match case
            Final=features


    
        x =Final# features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.out_proj(x)
       
        return x,Final # we also output the final representation


class SemiemoEncoder(FairseqDecoder):
    """RoBERTa encoder.

    Implements the :class:`~fairseq.models.FairseqDecoder` interface required
    by :class:`~fairseq.models.FairseqLanguageModel`.
    """

    def __init__(self, args, dictionary):
  
        super().__init__(dictionary)

        self.args = args


        self.sentence_encoder = TransformerMultiEncoder(     #This encodes all the 
            padding_idx=1,#dictionary.pad(), #Check!
            #vocab_size=0,#len(dictionary),#Check!
            #num_encoder_layers=args.encoder_layers,
            num_encoder_layers_cross=args.encoder_layers_cross,
            #embedding_dim=args.encoder_embed_dim,
            embedding_dim_text=args.encoder_embed_dim_t,
            embedding_dim_audio=args.encoder_embed_dim_a,
            #embedding_dim_video=args.encoder_embed_dim_v,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            #max_seq_len_text=args.max_positions_t,#This is the text
            #max_seq_len_audio=args.max_positions_a,
            #max_seq_len_video=args.max_positions_v,
            #num_segments=0,
            #encoder_normalize_before=True,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
            is_only_text=args.t_only,
            is_only_audio=args.a_only,
            #is_only_video=args.v_only,
            is_all_in=args.all_in,
            is_stack_up=args.stack_up

        )

          


    def forward(self, src_tokens, features_only=False, return_all_hiddens=False,aug=False, **unused):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the  LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states.
        """

   
        x, extra = self.extract_features(src_tokens, return_all_hiddens,aug) #Get the output after cross attentions and self attention mechanisms

       
        #Next track the loss function
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False,aug=False, **unused):  #This 
   
        
        inner_states, seq_rep = self.sentence_encoder(  #use the foewardfunction in TransformerMultiEncoder object
            src_tokens, last_state_only=not return_all_hiddens,is_aug=aug
        )

        return inner_states,seq_rep

    def max_positions(self):
        """Maximum length supported by the model."""
        return None #self.args.max_positions
        #return sys.maxsize



@register_model_architecture('semiemo', 'semiemo')
def semiemo_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 24)
    args.encoder_layers_cross = getattr(args, 'encoder_layers_cross', 1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024) 
    args.encoder_embed_dim_concat = getattr(args, 'encoder_embed_dim_concat',1792 )  #2048(1024 + 512 + 256)
    #args.encoder_embed_dim_concat = getattr(args, 'encoder_embed_dim_concat',1280 )  #2048(1024 + 512 + 256) #audio only
    args.encoder_embed_dim_t = getattr(args, 'encoder_embed_dim_t', 1024) 
    args.encoder_embed_dim_a = getattr(args, 'encoder_embed_dim_a', 768) 
    args.encoder_embed_dim_v = getattr(args, 'encoder_embed_dim_v', 256) 
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024) #previously this was 1024
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.0)
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)


