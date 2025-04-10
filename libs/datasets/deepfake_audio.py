import os
import json
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
import random,copy,glob
from .datasets import register_dataset
from .data_utils import truncate_feats
from ..utils import remove_duplicate_annotations


@register_dataset("deepfake_audio_inference")
class DeepFakeAudioDatasetInfer(Dataset):
    def __init__(
        self,
        is_training,      # if in training mode
        split,            # split, a tuple/list allowing concat of subsets
        sub_index,

        crop_ratio,
        default_fps,
        downsample_rate,
        audio_feat_folder, # folder for audio features
        audio_file_ext,
        num_classes,
        input_dim,
        audio_input_dim,
        feat_stride,
        num_frames,
        test_folder,
        trunc_thresh,
        max_seq_len,
        force_upsampling
    ):
        self.sub_index = sub_index
       
        self.audio_feat_folder= audio_feat_folder
       
        self.audio_file_ext=audio_file_ext
        self.test_folder = test_folder

        # anet uses fixed length features, make sure there is no downsampling
        self.force_upsampling = force_upsampling

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.audio_input_dim=audio_input_dim
       
        
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = {'Fake':0}
     
        # proposal vs action categories
        assert (num_classes == 1)
        self._get_test_infos()
        
        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'DeepFake_Audio',
            'tiou_thresholds': np.linspace(0.5, 0.95, 10), 
            'empty_label_ids': []
        }
        print("This testing has {} videos".format(len(self.data_list)))

        

    def _get_test_infos(self):
        data_list = []
        test_txt = os.path.join(self.test_folder, f"deepfake_test_sub{self.sub_index}.txt")
        with open(test_txt,'r') as f:
            lines = f.readlines()
        for line in lines:
            items = line.strip().split(',')
            data_list.append({'id':items[0],'duration':float(items[1])})

        self.data_list = data_list[45000:]
        # print('data_list:',len(data_list))


    def get_attributes(self):
        return self.db_attributes


    def __len__(self):
        return len(self.data_list)

        # 用一个单独的文件列表存，再shuffle
        

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]
        
        if self.audio_feat_folder is not None:
            audio_filename = os.path.join(self.audio_feat_folder, video_item['id'].replace('.mp4','.npy'))
            try:
                audio_feats = np.load(audio_filename)
                
            except ValueError as e:
                print(e)
                print(audio_filename)
               
        byola_fps = audio_feats.shape[0] / video_item['duration']
        # we support both fixed length features / variable length features
        # case 1: variable length features for training
        if self.feat_stride > 0 and (not self.force_upsampling): # 想走这一条
            # var length features
            feat_stride, num_frames = self.feat_stride, self.num_frames
            # only apply down sampling here
           
        # case 2: variable length features for input, yet resized for training
        
        elif self.feat_stride > 0 and self.force_upsampling: # 这条分支
            feat_stride = float(
                (audio_feats.shape[0] - 1) * self.feat_stride + self.num_frames
            ) / self.max_seq_len # 117/768 = 0.152 感觉像是当前特征的一行代表原始视频的多少FPS？ 不缩放的话默认是1FPS
            # center the features
            num_frames = feat_stride
        # case 3: fixed length features for input
        else:
            # deal with fixed length feature, recompute feat_stride, num_frames
            seq_len = audio_feats.shape[0]
            assert seq_len <= self.max_seq_len
            if self.force_upsampling:
                # reset to max_seq_len
                seq_len = self.max_seq_len
            feat_stride = video_item['duration'] * video_item['fps'] / seq_len
            # center the features
            num_frames = feat_stride

        feat_offset = 0.5 * num_frames / feat_stride 
        # print('feat_stride:',feat_stride) # 0.152
        # print('num_frames:',num_frames)# 0.152
        # print('feat_offset:',feat_offset) #0.5

            

        audio_feats = torch.from_numpy(np.ascontiguousarray(audio_feats.transpose()))
        if self.force_upsampling:
            resize_audio_feats = F.interpolate(
                audio_feats.unsqueeze(0),
                size=self.max_seq_len,
                mode='linear',
                align_corners=False
            )
            audio_feats = resize_audio_feats.squeeze(0)
     

        feats = audio_feats
        # feats = feats / torch.max(feats)
          
       
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T 用到了
                     'byola_fps'       : byola_fps, # eval用
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : num_frames}

        return data_dict



@register_dataset("deepfake_audio")
class DeepFakeAudioDataset(Dataset):
    def __init__(
        self,
        # ,
        is_training,      # if in training mode
        split,            # split, a tuple/list allowing concat of subsets
        feat_folder,      # folder for features
        audio_feat_folder, # folder for audio features
        train_txt,
        json_folder,        # json folder for annotations
        feat_stride,      # temporal stride of the feats
        num_frames,       # number of frames for each feat
        default_fps,      # default fps
        downsample_rate,  # downsample rate for feats 没看到这个
        max_seq_len,      # maximum sequence length during training
        trunc_thresh,     # threshold for truncate an action segment TODO 这是干啥的
        crop_ratio,       # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,        # input feat dim
        audio_input_dim, # input audio feat dim
        num_classes,      # number of action categories
        file_prefix,      # feature file prefix if any
        file_ext,         # feature file extension if any
        audio_file_ext,  # audio feature file extension if any
        force_upsampling  # force to upsample to max_seq_len
    ):
        # file path
        # assert os.path.exists(feat_folder) and os.path.exists(json_file)
        # assert isinstance(split, tuple) or isinstance(split, list)
        
        assert crop_ratio == None or len(crop_ratio) == 2
        self.train_txt = train_txt
        # self.train_txt = '/home/ubuntu/sn15_share_dir/av_deepfake/preprocess/processed_byola.txt'
        # print(f"======={feat_folder}=====")
        self.feat_folder = feat_folder
        self.audio_feat_folder= audio_feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.audio_file_ext=audio_file_ext
        self.json_folder = json_folder

        # anet uses fixed length features, make sure there is no downsampling
        self.force_upsampling = force_upsampling

        # split / training mode
        self.split = split
        print(f"======={split}=====")

        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.audio_input_dim=audio_input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = {'Fake':0}
        self.crop_ratio = crop_ratio
        # proposal vs action categories
        assert (num_classes == 1)
        self._get_train_jsons()#glob.glob(os.path.join('/home/ubuntu/shared-data/UMMAFormer/data/deepfake/train_metadata','*','*','*','real_video_fake_audio.json'))
        
        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'DeepFake_Audio',
            'tiou_thresholds': np.linspace(0.5, 0.95, 10), 
            'empty_label_ids': []
        }
        print("{} subset has {} videos".format(self.split,len(self.data_list)))

        self.byola_fps=12.5 # TODO 有些又是12.4几

    def _get_train_jsons(self):
        data_list = []
        with open(self.train_txt,'r') as f:
            lines = f.readlines()
        for line in lines:
            data_list.append(os.path.join(self.json_folder, line.strip().replace('.npy','.json')))
        self.data_list = data_list
        # print('data_list:',len(data_list))


    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        # print(f"================{json_file}===============")
        with open(json_file, 'r') as fid:
            value = json.load(fid)


        # fill in the db (immutable afterwards)
        # dict_db = tuple()
        # for value in json_db:
        # key =  os.path.splitext(os.path.basename(value['file']))[0]
        key = value['file'] # mp4的相对路径
        # skip the video if not in the split
        
        audio_bloya_feat_file = os.path.join(self.audio_feat_folder, os.path.dirname(key), os.path.basename(key).split('.')[0]+'.npy')
            # print(feat_file)
        
        duration = value['audio_frames'] / 16000
        # get fps if available
        if self.default_fps is not None:
            fps = self.default_fps
        elif 'fps' in value:
            fps = value['fps']
        elif 'video_frames' in value:
            fps = value['video_frames'] / duration # 这条分支
        else:
            assert False, "Unknown video FPS."
        # duration = value['duration']
        
        video_labels=0 if len(value['visual_fake_segments'])>0 else 1
        audio_labels=0 if len(value['audio_fake_segments'])>0 else 1
        av_labels = np.array([video_labels,audio_labels])
        # get annotations if available
        if ('fake_segments' in value) and (len(value['fake_segments']) > 0):
            valid_acts = value['fake_segments']
            num_acts = len(valid_acts)
            segments = np.zeros([num_acts, 2], dtype=np.float32)
            labels = np.zeros([num_acts, ], dtype=np.int64)
            for idx, act in enumerate(valid_acts):
                segments[idx][0] = act[0]
                segments[idx][1] = act[1]
                labels[idx] = 0
        else:
                segments = None
                labels = None
        dict_db = {'id': key,
                        'fps' : fps,
                        'duration' : duration,
                        'split': value['split'].lower(),
                        'segments' : segments,
                        'labels' : labels,
                        'av_labels': av_labels,
                        'bloya_feat_path':audio_bloya_feat_file
        }

        return dict_db

    def __len__(self):
        return len(self.data_list)

        # 用一个单独的文件列表存，再shuffle
        

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        json_file = self.data_list[idx]
        video_item = self._load_json_db(json_file)
        # load features
        
        
        if self.audio_feat_folder is not None:
            audio_filename = video_item['bloya_feat_path']
            try:
                audio_feats = np.load(audio_filename)
                
            except ValueError as e:
                print(e)
                print(audio_filename)
                # raise ValueError
                json_file = self.data_list[idx//2]
                video_item = self._load_json_db(json_file)
                audio_filename = video_item['bloya_feat_path']
                audio_feats = np.load(audio_filename)
            # 需要对这个进行截取
            audio_feats = audio_feats[:int(self.byola_fps*video_item['duration'])]
        

        # we support both fixed length features / variable length features
        # case 1: variable length features for training
        if self.feat_stride > 0 and (not self.force_upsampling): # 想走这一条
            # var length features
            feat_stride, num_frames = self.feat_stride, self.num_frames
            # only apply down sampling here
            if self.downsample_rate > 1:
                audio_feats = audio_feats[::self.downsample_rate, :]
                feat_stride = self.feat_stride * self.downsample_rate
        # case 2: variable length features for input, yet resized for training
        
        elif self.feat_stride > 0 and self.force_upsampling: # 这条分支
            feat_stride = float(
                (audio_feats.shape[0] - 1) * self.feat_stride + self.num_frames
            ) / self.max_seq_len # 117/768 = 0.152 感觉像是当前特征的一行代表原始视频的多少FPS？ 不缩放的话默认是1FPS
            # center the features
            num_frames = feat_stride
        # case 3: fixed length features for input
        else:
            # deal with fixed length feature, recompute feat_stride, num_frames
            seq_len = audio_feats.shape[0]
            assert seq_len <= self.max_seq_len
            if self.force_upsampling:
                # reset to max_seq_len
                seq_len = self.max_seq_len
            feat_stride = video_item['duration'] * video_item['fps'] / seq_len
            # center the features
            num_frames = feat_stride
        feat_offset = 0.5 * num_frames / feat_stride 
        # print('feat_stride:',feat_stride) # 0.152
        # print('num_frames:',num_frames)# 0.152
        # print('feat_offset:',feat_offset) #0.5

        
            
        if (self.audio_feat_folder is not None):
            audio_feats = torch.from_numpy(np.ascontiguousarray(audio_feats.transpose()))
            if self.force_upsampling:
                resize_audio_feats = F.interpolate(
                    audio_feats.unsqueeze(0),
                    size=self.max_seq_len,
                    mode='linear',
                    align_corners=False
                )
                audio_feats = resize_audio_feats.squeeze(0)
          
            # feats=torch.cat([feats,audio_feats],dim=0)
            feats = audio_feats
            # feats = feats / torch.max(feats) TODO 替换成正儿巴经从config读的
            # feats = torch.cat([audio_feats, audio_feats],dim=0) # TODO 假特征，为了直接用现有的ckpt，eval起来模型
            # feats = feats/400.
        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        
        if video_item['segments'] is not None:
            # feat_stride 的意义好像是将原始的frame_index映射到768长度下的新index
            # segments是[start_second, end_second]
            # print("segments origin:",video_item['segments'])
            # TODO 只有音频时，这儿好像不能用视频的fps
            segments = torch.from_numpy(
                video_item['segments'] * self.byola_fps / feat_stride - feat_offset
            )
            # print("segments in feature:",video_item['segments'] * video_item['fps'])
            # print("segments interp:(feat_stride,feat_offset)", segments,feat_stride,feat_offset)
            labels = torch.from_numpy(video_item['labels'])
            # for activity net, we have a few videos with a bunch of missing frames
            # here is a quick fix for training
            if self.is_training:
                vid_len = feats.shape[1] + feat_offset
                valid_seg_list, valid_label_list = [], []
                for seg, label in zip(segments, labels):
                    # print(seg,vid_len)
                    if seg[0] >= vid_len:
                        
                        # skip an action outside of the feature map
                        continue
                    # skip an action that is mostly outside of the feature map
                    ratio = (
                        (min(seg[1].item(), vid_len) - seg[0].item())
                        / (seg[1].item() - seg[0].item())
                    )
                    # print(ratio)
                    if ratio >= self.trunc_thresh:
                        valid_seg_list.append(seg.clamp(max=vid_len))
                        # some weird bug here if not converting to size 1 tensor
                        valid_label_list.append(label.view(1))
                segments = torch.stack(valid_seg_list, dim=0)
                # print("segments valid:",segments)
                labels = torch.cat(valid_label_list)
        else:
            segments, labels = None, None
            
        av_labels = torch.from_numpy(video_item['av_labels'])
        # print("!!! max[feats]:",torch.max(feats))
        # print("!! min(feats):",torch.min(feats))
        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T 用到了
                     'segments_time'   : video_item['segments'], # eval用
                     'segments'        : segments,   # N x 2 用到了 
                     'n_fakes'         : 0 if segments is None else segments.shape[0], # 为了eval新加的
                     'labels'          : labels,     # N 用到了
                     'av_labels'       : av_labels,
                     'fps'             : video_item['fps'],
                     'byola_fps'       : self.byola_fps, # eval用
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'split'           : video_item['split'],
                     'feat_num_frames' : num_frames}

        # no truncation is needed
        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
            )

        return data_dict

if __name__ == '__main__':
    deepfake = DeepFakeAudioDataset(is_training=True, split=['train'],
    feat_folder=None,
    audio_feat_folder='/home/ubuntu/shared-data/UMMAFormer/data/deepfake/train_wav',
    json_file=None,
    feat_stride=1,num_frames=1,default_fps=None,
    downsample_rate=0,max_seq_len=768,trunc_thresh=0.5,
    crop_ratio=[0.9,1.0],input_dim=0,audio_input_dim=2048,
    num_classes=1,file_prefix='rgb',file_ext='.npy',audio_file_ext=None,force_upsampling=True)
    data = deepfake[2]
    print(data['feats'].shape)
    print(data['segments'])
    print(data['labels']) #fake=0,real=1
    print(data['av_labels'])# [1,1]
    print(data['fps'])
