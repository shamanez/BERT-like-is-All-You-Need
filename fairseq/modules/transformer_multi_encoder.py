# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple
import numpy as np
from random import randrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
    PositionalEmbedding,
    TransformerSentenceEncoderLayer,
    TransformerMultiEncoderLayer,
)

import math



def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.v_proj.weight.data.normal_(mean=0.0, std=0.02)



class TransformerMultiEncoder(nn.Module):  #We might not need this part since we are already getting embeddings...
    
    def __init__(
        self,
        padding_idx: int,
        #vocab_size: int,
        #num_encoder_layers: int = 6,
        num_encoder_layers_cross: int = 6,
        #embedding_dim: int = 768,
        embedding_dim_text: int = 768,
        embedding_dim_audio: int = 768,
        #embedding_dim_video: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        #max_seq_len_text: int = 256,  
        #max_seq_len_audio: int = 256,
        #max_seq_len_video: int = 256,
        #num_segments: int = 2,
        #use_position_embeddings: bool = True,
        #is_start_AV_embeddings: bool = True, 
        #offset_positions_by_padding: bool = True, 
        #encoder_normalize_before: bool = False,
        apply_bert_init: bool = False,
        activation_fn: str = "relu",
        #learned_pos_embedding: bool = False, #we do not learn positonal embeddings
        #is_self_attention: bool =True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        embed_scale: float = None,
        #freeze_embeddings: bool = False,
        #n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        is_only_text: bool=False,
        is_only_audio: bool=False,
        #is_only_video: bool=False,
        is_all_in: bool=False,
        is_stack_up:bool=False, #telling whether we are to add more layers on top of SSL representation
    ) -> None:

        super().__init__()
        self.padding_idx = padding_idx
        #self.vocab_size = vocab_size
        self.dropout = dropout
        #self.max_seq_len_t = max_seq_len_text #text
        #self.max_seq_len_a = max_seq_len_audio #audio
        #self.max_seq_len_v = max_seq_len_video #video
        #self.embedding_dim = embedding_dim
        self.embedding_dim_t = embedding_dim_text
        self.embedding_dim_a = embedding_dim_audio
        #self.embedding_dim_v = embedding_dim_video
        #self.num_segments = num_segments
        #self.use_position_embeddings = use_position_embeddings
        #self.is_start_AV_embeddings=is_start_AV_embeddings
        self.apply_bert_init = apply_bert_init
        #self.learned_pos_embedding = learned_pos_embedding

        self.only_t=is_only_text
        self.only_a=is_only_audio
        #self.only_v=is_only_video
        self.all_in=is_all_in

        self.embed_scale = embed_scale

        self.stack_up=is_stack_up



        if self.stack_up:
         
            if (self.all_in) or (self.only_a and self.only_t):

            
                self.layers_ta = nn.ModuleList(  #Text to Audio (The query vector comes from the Text and Key-Value from the Audio)
                    [
                        TransformerMultiEncoderLayer(
                            embedding_dim=self.embedding_dim_t,#self.embedding_dim,
                            qdim=self.embedding_dim_t,
                            kdim=self.embedding_dim_a,
                            vdim=self.embedding_dim_a,
                            self_attention=False,
                            encoder_decoder_attention=True,
                            ffn_embedding_dim=ffn_embedding_dim,
                            num_attention_heads=num_attention_heads,
                            dropout=self.dropout,
                            attention_dropout=attention_dropout,
                            activation_dropout=activation_dropout,
                            activation_fn=activation_fn,
                            add_bias_kv=add_bias_kv,
                            add_zero_attn=add_zero_attn,
                            export=export,
                        )
                        for _ in range(num_encoder_layers_cross)
                    ]
                )

                self.layers_at = nn.ModuleList( #Audio to Text  (The query vector comes from the Audio and Key-Value from the Text)
                    [
                        TransformerMultiEncoderLayer(
                            embedding_dim=self.embedding_dim_a,#self.embedding_dim,
                            qdim=self.embedding_dim_a,
                            kdim=self.embedding_dim_t,
                            vdim=self.embedding_dim_t,
                            self_attention=False,
                            encoder_decoder_attention=True,
                            ffn_embedding_dim=ffn_embedding_dim,
                            num_attention_heads=num_attention_heads,
                            dropout=self.dropout,
                            attention_dropout=attention_dropout,
                            activation_dropout=activation_dropout,
                            activation_fn=activation_fn,
                            add_bias_kv=add_bias_kv,
                            add_zero_attn=add_zero_attn,
                            export=export,
                        )
                        for _ in range(num_encoder_layers_cross)
                    ]
                )

        
        
            # Apply initialization of model params after building the model
            if self.apply_bert_init:
                self.apply(init_bert_params)


     

    def forward(
        self,
        multi_modal_features: dict, #This tensor consist of output vectors from the three modalities 
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
        is_aug: bool =False
    ) -> Tuple[torch.Tensor, torch.Tensor]:


        last_states={}
        seq_rep={}

        tokens_only=multi_modal_features['raw_data']



        if (self.all_in) or (self.only_a):

            audio_features=multi_modal_features['Audio']
            raw_tokens_audio=tokens_only['audio']
            padding_mask_audio = raw_tokens_audio.eq(1)  #TEMP


                

            if is_aug: #as a data augmentation technique we  modify the padding mask

                aug_padding_mask_audio=[]
                aug_audio_cls_indexes=[]
                
                for i, n_in in enumerate(padding_mask_audio):
                    amount_not_masked=(len(n_in)- torch.sum(n_in, dim=0)).cpu().data.numpy()
                    init_mask=torch.cuda.FloatTensor(amount_not_masked).uniform_() > 0.3 #this is a new mask in the length of non padded elements
                    final_mask=torch.cat((n_in[0:1],init_mask[1:],n_in[amount_not_masked:]), 0) #add the new augmented padding mask
                    aug_padding_mask_audio.append(final_mask.unsqueeze(0))

                    aug_audio_cls_indexes.append(randrange(amount_not_masked))#aped possible CLS tokens



                padding_mask_audio=torch.cat(aug_padding_mask_audio)   #making the tensors
                aug_audio_cls_indexes=torch.LongTensor(aug_audio_cls_indexes) #making the tensors
        

                # aug_indx_cls_txt=randrange(amount_not_masked_t)  #this should use layer when using 
                # replace_cls_txt= padding_mask_audio[:,aug_indx_cls_txt]


            #applying the modified mask to the audio tokens
            if not padding_mask_audio.any():
                padding_mask_audio = None
            x_a =audio_features
            if self.embed_scale is not None:
                x_a = x_a*self.embed_scale

      
           
            j_aud_n=x_a[:, 0, :]
            seq_rep.update({'j_aud' : j_aud_n})

        

        if (self.all_in) or (self.only_t):
            raw_tokens_text=tokens_only['text']
            text_features=multi_modal_features['Text']
            padding_mask_text = raw_tokens_text.eq(1)  #Getting a mask for attention , 0s for elements with actual value

       
            if is_aug: #as a data augmentation technique we  modify the padding mask

                aug_padding_mask_text=[]
                aug_text_cls_indexes=[]
                
                for i, n_in in enumerate(padding_mask_text):
           
                    amount_not_masked=(len(n_in)- torch.sum(n_in, dim=0)).cpu().data.numpy()
                    init_mask=torch.cuda.FloatTensor(amount_not_masked).uniform_() > 0.1 #this is a new mask in the length of non padded elements
                    final_mask=torch.cat((n_in[0:1],init_mask[1:],n_in[amount_not_masked:]), 0) #add the new augmented padding mask
                    aug_padding_mask_text.append(final_mask.unsqueeze(0))

                    aug_text_cls_indexes.append(randrange(amount_not_masked))#aped possible CLS tokens


                padding_mask_text=torch.cat(aug_padding_mask_text)   #making the tensors
                aug_text_cls_indexes=torch.LongTensor(aug_text_cls_indexes) #making the tensors

                # aug_indx_cls_txt=randrange(amount_not_masked_t)  #this should use layer when using 
                # replace_cls_txt= padding_mask_audio[:,aug_indx_cls_txt]


             #applying the modified mask to the text tokens
            if not padding_mask_text.any():
                padding_mask_text = None
            x_t =text_features
            if self.embed_scale is not None:
                x_t = x_t*self.embed_scale 
                

            j_text=x_t[:, 0, :] #text embeddigs
            seq_rep.update({'j_text' : j_text})       

       

    
       
        if self.stack_up:
            if self.only_a or self.all_in: 
                #x_a = F.dropout(x_a, p=self.dropout, training=self.training)
                if padding_mask_audio is not None:
                    x_a = x_a* (1 - padding_mask_audio.unsqueeze(-1).type_as(x_a))
                x_a = x_a.transpose(0, 1)


            
            if self.only_t or self.all_in:
                if padding_mask_text is not None:
                    x_t =x_t* (1 - padding_mask_text.unsqueeze(-1).type_as(x_t))
                x_t = x_t.transpose(0, 1)
            


            if (self.all_in) or (self.only_t and self.only_a):
                x_ta=x_t[0,:,:].unsqueeze(0) #torch.Size([512, 1, 1024]) --->torch.Size([1, 1, 1024])
                for layer_ta in self.layers_ta:  #mask should be the key
                    x_ta,_=layer_ta(x_ta,x_a,x_a, self_attn_padding_mask=padding_mask_audio)
           
                
                x_at=x_a[0,:,:].unsqueeze(0)
                for layer_at in self.layers_at:
                    x_at,_=layer_at(x_at,x_t,x_t, self_attn_padding_mask=padding_mask_text)

                x_ta = x_ta.transpose(0, 1)
                x_at = x_at.transpose(0, 1)

                ta_rep = x_ta[:, 0, :]
                at_rep = x_at[:, 0, :]

                seq_rep.update({'t2a_r' : ta_rep})
                seq_rep.update({'a2t_r' : at_rep})

               
        return last_states, seq_rep
