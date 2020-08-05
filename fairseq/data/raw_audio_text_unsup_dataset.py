
import os
import numpy as np
import sys
import torch

from .import FairseqDataset
import time

class RawAudioTextUnsupDataset(FairseqDataset):

    def __init__(self, base_path,data_args,data_split,dataset_type, sample_rate, max_sample_size=None, min_sample_size=None,
                 shuffle=True):
        super().__init__()


        
        #we do not need the datasplit here
        self.data_args=data_args


        self.fnames_audio = []
        self.fnames_text = []
        self.sizes = []

  

        self.labels = {}

        self.audio_sizes = {}
        self.text_sizes = {}

        

        self.max_sample_size = max_sample_size if max_sample_size is not None else sys.maxsize
        self.min_sample_size = min_sample_size if min_sample_size is not None else self.max_sample_size
        self.base_manifest_path = base_path
        self.split = dataset_type#data_split




        manifest_audio = os.path.join(self.base_manifest_path, '{}.tsv'.format(self.split+"_a"))
        manifest_text = os.path.join(self.base_manifest_path, '{}.tsv'.format(self.split+"_t"))
        manifest_size = os.path.join(self.base_manifest_path, '{}.tsv'.format(self.split+"_size"))


  

        with open(manifest_size, 'r') as f_s :
          
            for line_l in f_s:

                items_s = line_l.strip().split(',')

                self.text_sizes[items_s[0].strip()] = items_s[1].strip()
                self.audio_sizes[items_s[0].strip()] = items_s[2].strip() #for the sentiment use 2 from the list else 1



     
        inter_n=0
        with open(manifest_audio, 'r') as f_a, open(manifest_text, 'r') as f_t:#, open(manifest_label, 'r') as f_l:


            # self.root_dir_a =os.path.join(self.data_args.data_raw , self.split ,'audio_token')     #f_a.readline().strip()
            # self.root_dir_t =os.path.join(self.data_args.data_raw , self.split ,'text')   #f_t.readline().strip()

            # self.root_dir_a =os.path.join(self.data_args.data_raw , 'train' ,'audio_token')     #f_a.readline().strip()
            # self.root_dir_t =os.path.join(self.data_args.data_raw , 'train' ,'text')   #f_t.readline().strip()

            self.root_dir_a =os.path.join('/hpc/gsir059/INTERSPEECH/MOSI-SEMI/data-bin/meld' , 'train' ,'audio_token')     #f_a.readline().strip()
            self.root_dir_t =os.path.join('/hpc/gsir059/INTERSPEECH/MOSI-SEMI/data-bin/meld' , 'train' ,'text')   #f_t.readline().strip()

            #"/hpc/gsir059/INTERSPEECH/MOSI-SEMI/data-bin/meld/train"


            for line_a, line_t in zip(f_a,f_t):#,f_l):, line_l

           
                items_a = line_a.strip().split('\t')
                items_t = line_t.strip().split('\t')

                assert items_a[0].split('.')[0] == items_t[0].split('.')[0] , "misalignment of data"

                self.fnames_audio.append(items_a[0].replace('.wav','.txt'))
                self.fnames_text.append(items_t[0])
                self.sizes.append(int(self.audio_sizes.get(items_a[0].split('.')[0])))
           

        self.shuffle = shuffle

    def __getitem__(self, index):

    
        audio_file = self.fnames_audio[index]
        text_file = self.fnames_text[index]
       

        fname_a = os.path.join(self.root_dir_a, audio_file)
        fname_t = os.path.join(self.root_dir_t, text_file)

     
        file_name = audio_file.replace('.txt','')
     
        assert file_name == text_file.replace('.txt',''), "not all file ids match"

  
        # Text data (Roberta Tokens)
        with open(fname_t, 'r') as f:
            words = []
            for line in f:
                words.extend(line.strip().split('\t'))
        tokensized_text = [int(word) for word in words]
        tokensized_text = torch.from_numpy(np.array(tokensized_text))

        # Text data (Roberta Tokens)
        with open(fname_a, 'r') as f:
            words = []
            for line in f:
                words.extend(line.strip().split('\t'))
        tokensized_audio = [int(word) for word in words]
        tokensized_audio = torch.from_numpy(np.array(tokensized_audio))


   
        return {
            'id': index,
            'text': tokensized_text,
            'audio_token':tokensized_audio,
        }

    def __len__(self): #Training dataset size
        return len(self.fnames_audio)

    def collate_tokens(self, values, pad_idx, max_target_value,eos_idx=None, left_pad=False, move_eos_to_beginning=False):
        """Convert a list of 1d tensors into a padded 2d tensor."""

        size =max_target_value#max(v.size(0) for v in values) #Here the size can be fixed as 512
        res = values[0].new(len(values), size).fill_(pad_idx)

     
        def copy_tensor(src, dst):

           
            if src.numel()>dst.numel():
                clip_src=src[:dst.numel()-1]
                src=torch.cat((clip_src, torch.tensor([2])), 0)

        
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                assert src[-1] == eos_idx
                dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)
        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
        return res

    def collate_audio_tokens(self, values, pad_idx,max_target_value, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
        """Convert a list of 1d tensors into a padded 2d tensor."""

        size = max_target_value#max(v.size(0) for v in values) #Here the size can be fixed as 512
        res = values[0].new(len(values), size).fill_(pad_idx)

    
        
        def copy_tensor(src, dst):
            if src.numel()>dst.numel():
                clip_src=src[:dst.numel()-1]
                src=torch.cat((clip_src, torch.tensor([2])), 0)
               
      
            assert dst.numel() == src.numel()

            if move_eos_to_beginning:
                assert src[-1] == eos_idx
                dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)
        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
        return res



    def collater(self, samples):
        
        if len(samples) == 0:
            return {}

        ####################################################################
        #collater for text chunks        
        #############################################
        sources_text = [s['text'] for s in samples]
        sizes_text = [len(s) for s in sources_text]
        max_target_size_t = min(max(sizes_text), 512) # max text token seq length


        collated_text = self.collate_tokens(sources_text, 1,max_target_size_t) #1 is the padding index


    
        ####################################################################
        #collater  for audio token chunks        
        #############################################
        sources_audio_tokens = [s['audio_token'] for s in samples]
        sizes_audio = [len(s) for s in sources_audio_tokens]
        max_target_size_a = min(max(sizes_audio), 2048) # max audio token seq length

        collated_audio_tokens = self.collate_audio_tokens(sources_audio_tokens, 1,max_target_size_a) #1 is the padding index


        return {
            'id': torch.LongTensor([s['id'] for s in samples]),
            'net_input': {
                'audio': collated_audio_tokens, 
                'text': collated_text, 
            }
        }


    def get_dummy_batch(
            self, num_tokens, max_positions, src_lne=2048, tgt_len=128,
    ):
        """Return a dummy batch with a given number of tokens."""
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            src_len = min(src_len, max_positions)
        bsz = num_tokens // src_len
        
        return self.collater([
            {
                'id': i,
                'audio': torch.rand(self.channels, self.timeDepth, self.xSize, self.ySize),
                'text': torch.rand(src_len),
                'video' : torch.rand(src_len)
            }
            for i in range(bsz)
            ])

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):

        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return min(self.sizes[index], self.max_sample_size)

    # def ordered_indices(self):  #Need to customize this
    #     """Return an ordered list of indices. Batches will be constructed based
    #     on this order."""

    #     if self.shuffle:  #Shuffeling the training dataset
    #         order = np.random.permutation(len(self))
    #     else:
    #         order = np.arange(len(self))
 
    #     return order


    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)
