from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pickle5 as pickle
import pandas as pd
from collections import defaultdict
import json
import random
import h5py
from tqdm import tqdm
from scipy import sparse
import h5py
import glob

class MSRVTT_Caption_DataLoader(Dataset):
    """MSRVTT train dataset loader."""
    def __init__(
            self,
            json_path,
            features_path,
            inference_path,
            tokenizer,
            cliptokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            split_type="train",
            use_raw_video=False,
    ):
        self.data = json.load(open(json_path, 'r'))
        self.feature_dict = pickle.load(open(features_path, 'rb'))
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.cliptokenizer = cliptokenizer
        self.use_raw_video = use_raw_video




        self.feature_size = self.feature_dict[next(iter(self.feature_dict))].shape
        self.inference_path = inference_path

        assert split_type in ["train", "val", "test"]
        # Train: video0 : video6512 (6513)
        # Val: video6513 : video7009 (497)
        # Test: video7010 : video9999 (2990)
        if not self.use_raw_video:
            video_ids = [self.data['videos'][idx]['video_id'] for idx in range(len(self.data['videos']))]
        split_dict = {"train": video_ids[:6513], "val": video_ids[6513:6513 + 497], "test": video_ids[6513 + 497:]}
        choiced_video_ids = split_dict[split_type]

        if self.inference_path is not None:
            inference_files = glob.glob(self.inference_path + "/*")
            inference_files = [x.split("/")[-1].split(".mp4")[0] for x in inference_files]
            choiced_video_ids = [x for x in choiced_video_ids if x in inference_files]

        self.sample_len = 0
        self.sentences_dict = {}
        self.video_sentences_dict = defaultdict(list)
        
        if split_type == "train":  # expand all sentence to train
            for itm in self.data['sentences']:
                if itm['video_id'] in choiced_video_ids:
                    self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
                    self.video_sentences_dict[itm['video_id']].append(itm['caption'])
        elif split_type == "val" or split_type == "test":
            for itm in self.data['sentences']:
                if itm['video_id'] in choiced_video_ids:
                    self.video_sentences_dict[itm['video_id']].append(itm['caption'])
            for vid in choiced_video_ids:
                self.sentences_dict[len(self.sentences_dict)] = (vid, self.video_sentences_dict[vid][0])
        else:
            raise NotImplementedError

        self.sample_len = len(self.sentences_dict)

        

            
    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)

        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = self.cliptokenizer.tokenize(caption)
            words = ["[CLS]"] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + ["[SEP]"]

            input_ids = self.cliptokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

            pairs_text[i] = np.array(input_ids)

            # For generate captions
            if caption is not None:
                caption_words = self.tokenizer.tokenize(caption)
            else:
                caption_words = self._get_single_text(video_id)
            if len(caption_words) > total_length_with_CLS:
                caption_words = caption_words[:total_length_with_CLS]
            input_caption_words = ["[CLS]"] + caption_words
            output_caption_words = caption_words + ["[SEP]"]

            # For generate captions
            input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
            output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)
            decoder_mask = [1] * len(input_caption_ids)
            while len(input_caption_ids) < self.max_words:
                input_caption_ids.append(0)
                output_caption_ids.append(0)
                decoder_mask.append(0)
            assert len(input_caption_ids) == self.max_words
            assert len(output_caption_ids) == self.max_words
            assert len(decoder_mask) == self.max_words

            pairs_input_caption_ids[i] = np.array(input_caption_ids)
            pairs_output_caption_ids[i] = np.array(output_caption_ids)
            pairs_decoder_mask[i] = np.array(decoder_mask)

        return pairs_text, pairs_mask,pairs_segment, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, choice_video_ids


    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_video(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        video = np.zeros((len(choice_video_ids), self.max_frames, self.feature_size[-1]), dtype=np.float)
        for i, video_id in enumerate(choice_video_ids):
            video_slice = self.feature_dict[video_id]

            if self.max_frames < video_slice.shape[0]:
                video_slice = video_slice[:self.max_frames]

            slice_shape = video_slice.shape
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
            if len(video_slice) < 1:
                print("video_id: {}".format(video_id))
            else:
                video[i][:slice_shape[0]] = video_slice

            for i, v_length in enumerate(max_video_length):
                video_mask[i][:v_length] = [1] * v_length

        return video, video_mask, np.array([]), np.array([])

    def _get_rawvideo(self, choice_video_ids):

        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)


        video = np.zeros((len(choice_video_ids),self.max_frames, self.feature_size[1], self.feature_size[2], self.feature_size[3]), dtype=np.float)

        max_video_length = [0] * len(choice_video_ids)
        

        for i, video_id in enumerate(choice_video_ids):

            video_slice = self.feature_dict[video_id]['video']
            
            

            if self.max_frames < video_slice.shape[0]:
                video_slice = video_slice[:self.max_frames]

            slice_shape = video_slice.shape
            video_mask[i][:slice_shape[0]] = self.feature_dict[video_id]['video_mask']
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
            if len(video_slice) < 1:
                print("video_id: {}".format(video_id))
            else:
                video[i][:slice_shape[0]] = video_slice

            

        
        return video, video_mask, np.array([]), np.array([])

    def __getitem__(self, idx):
        video_id, caption = self.sentences_dict[idx]

        video_id, caption = self.sentences_dict[idx]
        pairs_text, pairs_mask, pairs_segment, \
        pairs_input_caption_ids, pairs_decoder_mask, \
        pairs_output_caption_ids, choice_video_ids = self._get_text(video_id, caption)

        if not self.use_raw_video:

            video, video_mask, masked_video, video_labels_index = self._get_video(choice_video_ids)
        else:
            video, video_mask, masked_video, video_labels_index = self._get_rawvideo(choice_video_ids)
        




        return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids



