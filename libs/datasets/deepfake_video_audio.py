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
        video_input_dim,
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


@register_dataset("deepfake_video_audio_inference")
class DeepFakeVideoAudioDatasetInfer(Dataset):
    def __init__(
        self,
        is_training,      # if in training mode
        split,            # split, a tuple/list allowing concat of subsets
        sub_index,
        crop_ratio,
        default_fps,
        downsample_rate,
        video_feat_folder,
        audio_feat_folder, # folder for audio features
        audio_file_ext,
        num_classes,
        input_dim,
        video_input_dim,
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
        self.video_feat_folder = video_feat_folder
       
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
        # test_txt = '/home/ubuntu/sn15_share_dir/av_deepfake/train_results/not_prcessed_test.txt'
        # test_txt = os.path.join(self.test_folder, "eval_test.txt")
        with open(test_txt,'r') as f:
            lines = f.readlines()
        for line in lines:
            items = line.strip().split(',')
            data_list.append({'id':items[0],'duration':float(items[1])})

        self.data_list = data_list
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
        
        if self.video_feat_folder is not None:
            video_filename = os.path.join(self.video_feat_folder, video_item['id'].replace('.mp4','.npy'))
            try:
                video_feats = np.load(video_filename)
                # print('original video feats shape:',video_feats.shape)
                
            except ValueError as e:
                print(e)
                print(video_filename)

        fps = video_feats.shape[0] / video_item['duration'] 

        if self.audio_feat_folder is not None:
            audio_filename = os.path.join(self.audio_feat_folder, video_item['id'].replace('.mp4','.npy'))
            try:
                audio_feats = np.load(audio_filename)
                
            except ValueError as e:
                print(e)
                print(audio_filename)
            # print('audio feature leng:',audio_feats.shape[0])
            # print('computed cut length:',int(50*video_item['duration']-0.817))
            audio_feats = audio_feats[:int(50*video_item['duration']-0.817)]
               
      
        # we support both fixed length features / variable length features
        # case 1: variable length features for training
        if self.feat_stride > 0 and (not self.force_upsampling): # 想走这一条
            # var length features
            feat_stride, num_frames = self.feat_stride, self.num_frames
            # only apply down sampling here
           
        # case 2: variable length features for input, yet resized for training
        
        elif self.feat_stride > 0 and self.force_upsampling: # 这条分支
            feat_stride = float(
                (video_feats.shape[0] - 1) * self.feat_stride + self.num_frames
            ) / self.max_seq_len # 117/768 = 0.152 感觉像是当前特征的一行代表原始视频的多少FPS？ 不缩放的话默认是1FPS
            # center the features
            num_frames = feat_stride
        # case 3: fixed length features for input
        else:
            raise RuntimeError('not implemented')

        feat_offset = 0.5 * num_frames / feat_stride 
        # print('feat_stride:',feat_stride) # 0.152
        # print('num_frames:',num_frames)# 0.152
        # print('feat_offset:',feat_offset) #0.5

        video_feats = torch.from_numpy(np.ascontiguousarray(video_feats.transpose())) #现在变成了shape[-1]是时间

        # resize the features if needed
        if (video_feats.shape[-1] != self.max_seq_len) and self.force_upsampling: # 同样是问题1，但是涉及到多模态，或者是不同模型输出的特征做拼接时，不同特征的HZ不一样，肯定需要把形状调整，这么直接interpolate是最佳方案吗？
            resize_feats = F.interpolate( # 测试发现，interpolate会直接改变特征内的值，例如[1,2,3,4]-->inter to 8 size-->[1.0000, 1.2500, 1.7500, 2.2500, 2.7500, 3.2500, 3.7500, 4.0000]
                video_feats.unsqueeze(0),
                size=self.max_seq_len,
                mode='linear',
                align_corners=False
            )
            video_feats = resize_feats.squeeze(0)    

        audio_feats = torch.from_numpy(np.ascontiguousarray(audio_feats.transpose()))
        if self.force_upsampling:
            resize_audio_feats = F.interpolate(
                audio_feats.unsqueeze(0),
                size=self.max_seq_len,
                mode='linear',
                align_corners=False
            )
            audio_feats = resize_audio_feats.squeeze(0)
     

        # feats = audio_feats
        feats=torch.cat([video_feats,audio_feats],dim=0)
        # feats = feats / torch.max(feats)
          
       
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T 用到了
                     'fps'             : fps, # eval用
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : num_frames}

        return data_dict


@register_dataset("deepfake_video_audioEmoBYOLA_inference")
class DeepFakeVideoAudioDatasetInfer3(Dataset):
    def __init__(
        self,
        is_training,      # if in training mode
        split,            # split, a tuple/list allowing concat of subsets
        sub_index,
        crop_ratio,
        default_fps,
        downsample_rate,
        video_feat_folder,
        audio_feat_folder, # folder for audio features
        audio_byola_feat_folder,
        audio_emo_feat_folder,
        audio_file_ext,
        num_classes,
        input_dim,
        video_input_dim,
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
        self.video_feat_folder = video_feat_folder
        self.audio_byola_feat_folder = audio_byola_feat_folder
        self.audio_emo_feat_folder = audio_emo_feat_folder

       
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

        self.byola_fps=12.497 
        self.emotion_fps = 50 

        

    def _get_test_infos(self):
        data_list = []
        test_txt = os.path.join(self.test_folder, f"deepfake_test_sub{self.sub_index}.txt")
        # test_txt = '/home/ubuntu/sn15_share_dir/av_deepfake/train_results/not_prcessed_test.txt'
        # test_txt = os.path.join(self.test_folder, "eval_test.txt")
        with open(test_txt,'r') as f:
            lines = f.readlines()
        for line in lines:
            items = line.strip().split(',')
            data_list.append({'id':items[0],'duration':float(items[1])})

        self.data_list = data_list
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
        
        if self.video_feat_folder is not None:
            video_filename = os.path.join(self.video_feat_folder, video_item['id'].replace('.mp4','.npy'))
            try:
                video_feats = np.load(video_filename)
                # print('original video feats shape:',video_feats.shape)
                
            except ValueError as e:
                print(e)
                print(video_filename)

        fps = video_feats.shape[0] / video_item['duration'] 

        if self.audio_byola_feat_folder is not None:
            audio_filename1 = os.path.join(self.audio_byola_feat_folder, video_item['id'].replace('.mp4','.npy'))
            try:
                audio_feats1 = np.load(audio_filename1)
                
            except ValueError as e:
                print(e)
                print(audio_filename1)
        
        if self.audio_emo_feat_folder is not None:
            audio_filename2 = os.path.join(self.audio_emo_feat_folder, video_item['id'].replace('.mp4','.npy'))
            try:
                audio_feats2 = np.load(audio_filename2)
                
            except ValueError as e:
                print(e)
                print(audio_filename2)       
            
        # 需要对这个进行截取
        audio_feats1 = audio_feats1[:int(self.byola_fps*video_item['duration']-0.3657)]
        audio_feats2 = audio_feats2[:int(self.emotion_fps*video_item['duration']-0.817)]
               

        # we support both fixed length features / variable length features
        # case 1: variable length features for training
        if self.feat_stride > 0 and (not self.force_upsampling): # 想走这一条
            # var length features
            feat_stride, num_frames = self.feat_stride, self.num_frames
            # only apply down sampling here
           
        # case 2: variable length features for input, yet resized for training
        
        elif self.feat_stride > 0 and self.force_upsampling: # 这条分支
            feat_stride = float(
                (video_feats.shape[0] - 1) * self.feat_stride + self.num_frames
            ) / self.max_seq_len # 117/768 = 0.152 感觉像是当前特征的一行代表原始视频的多少FPS？ 不缩放的话默认是1FPS
            # center the features
            num_frames = feat_stride
        # case 3: fixed length features for input
        else:
            raise RuntimeError('not implemented')

        feat_offset = 0.5 * num_frames / feat_stride 
        # print('feat_stride:',feat_stride) # 0.152
        # print('num_frames:',num_frames)# 0.152
        # print('feat_offset:',feat_offset) #0.5

        video_feats = torch.from_numpy(np.ascontiguousarray(video_feats.transpose())) #现在变成了shape[-1]是时间
       
        # resize the features if needed
        if (video_feats.shape[-1] != self.max_seq_len) and self.force_upsampling: # 同样是问题1，但是涉及到多模态，或者是不同模型输出的特征做拼接时，不同特征的HZ不一样，肯定需要把形状调整，这么直接interpolate是最佳方案吗？
            resize_feats = F.interpolate( # 测试发现，interpolate会直接改变特征内的值，例如[1,2,3,4]-->inter to 8 size-->[1.0000, 1.2500, 1.7500, 2.2500, 2.7500, 3.2500, 3.7500, 4.0000]
                video_feats.unsqueeze(0),
                size=self.max_seq_len,
                mode='linear',
                align_corners=False
            )
            video_feats = resize_feats.squeeze(0)

        # resize the features if needed
        if (self.audio_byola_feat_folder is not None):
            audio_feats1 = torch.from_numpy(np.ascontiguousarray(audio_feats1.transpose()))
            if self.force_upsampling:
                resize_audio_feats1 = F.interpolate(
                    audio_feats1.unsqueeze(0),
                    size=self.max_seq_len,
                    mode='linear',
                    align_corners=False
                )
                audio_feats1 = resize_audio_feats1.squeeze(0)

                # print('interp [audio_feats]:',audio_feats.shape)
        if (self.audio_emo_feat_folder is not None):
            audio_feats2 = torch.from_numpy(np.ascontiguousarray(audio_feats2.transpose()))
            if self.force_upsampling:
                resize_audio_feats2 = F.interpolate(
                    audio_feats2.unsqueeze(0),
                    size=self.max_seq_len,
                    mode='linear',
                    align_corners=False
                )
                audio_feats2 = resize_audio_feats2.squeeze(0)
            # feats=torch.cat([feats,audio_feats],dim=0)
        
        feats=torch.cat([video_feats,audio_feats1,audio_feats2],dim=0)
        # feats = feats / torch.max(feats)
          
       
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T 用到了
                     'fps'             : fps, # eval用
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : num_frames}

        return data_dict


@register_dataset("deepfake_video_audioBYOLA_inference")
class DeepFakeVideoAudioDatasetInfer2(Dataset):
    def __init__(
        self,
        is_training,      # if in training mode
        split,            # split, a tuple/list allowing concat of subsets
        sub_index,
        crop_ratio,
        default_fps,
        downsample_rate,
        video_feat_folder,
        audio_feat_folder, # folder for audio features
        audio_file_ext,
        num_classes,
        input_dim,
        video_input_dim,
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
        self.video_feat_folder = video_feat_folder
       
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

        self.byola_fps=12.497 
        # self.emotion_fps = 50 

        

    def _get_test_infos(self):
        data_list = []
        # test_txt = os.path.join(self.test_folder, f"deepfake_test_sub{self.sub_index}.txt")
        test_txt = '/home/ubuntu/sn15_share_dir/av_deepfake/train_results/not_prcessed_test.txt'
        # test_txt = os.path.join(self.test_folder, "eval_test.txt")
        with open(test_txt,'r') as f:
            lines = f.readlines()
        for line in lines:
            items = line.strip().split(',')
            data_list.append({'id':items[0],'duration':float(items[1])})

        self.data_list = data_list
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
        
        if self.video_feat_folder is not None:
            video_filename = os.path.join(self.video_feat_folder, video_item['id'].replace('.mp4','.npy'))
            try:
                video_feats = np.load(video_filename)
                # print('original video feats shape:',video_feats.shape)
                
            except ValueError as e:
                print(e)
                print(video_filename)

        fps = video_feats.shape[0] / video_item['duration'] 

        if self.audio_feat_folder is not None:
            audio_filename = os.path.join(self.audio_feat_folder, video_item['id'].replace('.mp4','.npy'))
            try:
                audio_feats = np.load(audio_filename)
                
            except ValueError as e:
                print(e)
                print(audio_filename)
        
              
            
            # 需要对这个进行截取
        audio_feats = audio_feats[:int(self.byola_fps*video_item['duration']-0.3657)]
       
               

        # we support both fixed length features / variable length features
        # case 1: variable length features for training
        if self.feat_stride > 0 and (not self.force_upsampling): # 想走这一条
            # var length features
            feat_stride, num_frames = self.feat_stride, self.num_frames
            # only apply down sampling here
           
        # case 2: variable length features for input, yet resized for training
        
        elif self.feat_stride > 0 and self.force_upsampling: # 这条分支
            feat_stride = float(
                (video_feats.shape[0] - 1) * self.feat_stride + self.num_frames
            ) / self.max_seq_len # 117/768 = 0.152 感觉像是当前特征的一行代表原始视频的多少FPS？ 不缩放的话默认是1FPS
            # center the features
            num_frames = feat_stride
        # case 3: fixed length features for input
        else:
            raise RuntimeError('not implemented')

        feat_offset = 0.5 * num_frames / feat_stride 
        # print('feat_stride:',feat_stride) # 0.152
        # print('num_frames:',num_frames)# 0.152
        # print('feat_offset:',feat_offset) #0.5

        video_feats = torch.from_numpy(np.ascontiguousarray(video_feats.transpose())) #现在变成了shape[-1]是时间

        # resize the features if needed
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
        feats=torch.cat([video_feats,audio_feats],dim=0)
        # feats = feats / torch.max(feats)
          
       
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T 用到了
                     'fps'             : fps, # eval用
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : num_frames}

        return data_dict



@register_dataset("deepfake_video_audio")
class DeepFakeVideoAudioDataset(Dataset):
    def __init__(
        self,
        # ,
        is_training,      # if in training mode
        split,            # split, a tuple/list allowing concat of subsets
        video_feat_folder,      # folder for features
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
        video_input_dim,        # input feat dim
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
        self.video_feat_folder = video_feat_folder
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
        self.video_input_dim = video_input_dim
        self.audio_input_dim=audio_input_dim
        self.default_fps = None
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = {'Fake':0}
        self.crop_ratio = crop_ratio
        # proposal vs action categories
        assert (num_classes == 1)
        self._get_train_list()#glob.glob(os.path.join('/home/ubuntu/shared-data/UMMAFormer/data/deepfake/train_metadata','*','*','*','real_video_fake_audio.json'))
        
        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'DeepFake_Audio',
            'tiou_thresholds': np.linspace(0.5, 0.95, 10), 
            'empty_label_ids': []
        }
        print("{} subset has {} videos".format(self.split,len(self.data_list)))
        print("============init deepfake_video_audio===dataest==========")

        # self.byola_fps=12.5 # TODO 有些又是12.4几
        self.emotion_fps = 50 

    def _get_train_list(self):
        data_list = []
        with open(self.train_txt,'r') as f:
            lines = f.readlines()
        for line in lines:
            data_list.append(line.strip())
        # random.shuffle(data_list)
        self.data_list = data_list
        print('data_list:',len(data_list))


    def get_attributes(self):
        return self.db_attributes

    

    def get_av_labels(self,json_filename):# 实际上返回的是video_audio_label
        if 'fake_video_real_audio.json' in json_filename:
            return [0,1]
        elif 'fake_video_fake_audio.json' in json_filename:
            return [0,0]
        elif 'real_video_fake_audio.json' in json_filename:
            return [1,0]
        elif 'real.json' in json_filename:
            return [1,1]
        else:
            raise ValueError(f"[Error] in get_av_labels() json file {json_filename} is incorrect")


    def _load_json_db(self, json_file):
        # load database and select the subset
        # print(f"================{json_file}===============")
        with open(os.path.join(self.json_folder,json_file), 'r') as fid:
            value = json.load(fid)


        # fill in the db (immutable afterwards)
        # dict_db = tuple()
        # for value in json_db:
        # key =  os.path.splitext(os.path.basename(value['file']))[0]
      
        
        
        duration = value['audio_frames'] / 16000
        # get fps if available
        if self.default_fps is not None:
            fps = self.default_fps
        elif 'fps' in value:
            fps = value['fps']
        elif 'video_frames' in value:
            fps = value['video_frames'] / duration # 这条分支
            # print('original fps:',fps)
        else:
            assert False, "Unknown video FPS."
        # duration = value['duration']
        # print('original fps:',fps)
        
       
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
        dict_db = {
                    'fps' : fps,
                    'duration' : duration,
                    'split': value['split'].lower(),
                    'segments' : segments,
                    'labels' : labels,
                        
                       
        }

        return dict_db

    def __len__(self):
        return len(self.data_list)

        # 用一个单独的文件列表存，再shuffle
        
    def error_item(self,idx):
        json_rela_path = self.data_list[idx//2]
        av_label_list = self.get_av_labels(json_rela_path)
        video_item = self._load_json_db(json_rela_path)
        feature_rela_path = json_rela_path.replace('.json','.npy')
        video_filename = os.path.join(self.video_feat_folder, feature_rela_path)
        video_feats = np.load(video_filename)
        audio_filename = os.path.join(self.audio_feat_folder, feature_rela_path)
        audio_feats = np.load(audio_filename)
        return av_label_list,video_feats,audio_feats

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        json_rela_path = self.data_list[idx]
        av_label_list = self.get_av_labels(json_rela_path)
        
        # print('json_file:',json_file)
        video_item = self._load_json_db(json_rela_path)

        feature_rela_path = json_rela_path.replace('.json','.npy')
        # load features
        if self.video_feat_folder is not None:
            video_filename = os.path.join(self.video_feat_folder, feature_rela_path)
            try:
                video_feats = np.load(video_filename)
                
            except ValueError as e:
                print(e)
                print(video_filename)
            # TODO 弄清楚这个需不需要截取
            # print("origin video_feats.shape:",video_feats.shape) # 需要shape[0]是时间 # (306,256)
            
        if self.audio_feat_folder is not None:
            audio_filename = os.path.join(self.audio_feat_folder, feature_rela_path)
            try:
                audio_feats = np.load(audio_filename)
                
            except ValueError as e:
                print(e)
                print(audio_filename)
                # 最好不中断训练，全部重新取个特征
                av_label_list,video_feats,audio_feats = self.error_item(idx)
               
            
            # 需要对这个进行截取
            audio_feats = audio_feats[:int(self.emotion_fps*video_item['duration']-0.817)]
            # print("resized audio_feats.shape:",audio_feats.shape) 
        

        # we support both fixed length features / variable length features
        # case 1: variable length features for training
        if self.feat_stride > 0 and (not self.force_upsampling): # 想走这一条
            # var length features
            feat_stride, num_frames = self.feat_stride, self.num_frames
            # only apply down sampling here
            if self.downsample_rate > 1:
                video_feats = video_feats[::self.downsample_rate, :]
                feat_stride = self.feat_stride * self.downsample_rate
        # case 2: variable length features for input, yet resized for training
        
        elif self.feat_stride > 0 and self.force_upsampling: # 这条分支
            feat_stride = float(
                (video_feats.shape[0] - 1) * self.feat_stride + self.num_frames
            ) / self.max_seq_len # 117/768 = 0.152 感觉像是当前特征的一行代表原始视频的多少FPS？ 不缩放的话默认是1FPS
            # center the features
            num_frames = feat_stride
        # case 3: fixed length features for input
        else:
            # deal with fixed length feature, recompute feat_stride, num_frames
            seq_len = video_feats.shape[0]
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

        
        video_feats = torch.from_numpy(np.ascontiguousarray(video_feats.transpose())) #现在变成了shape[-1]是时间

        # resize the features if needed
        if (video_feats.shape[-1] != self.max_seq_len) and self.force_upsampling: # 同样是问题1，但是涉及到多模态，或者是不同模型输出的特征做拼接时，不同特征的HZ不一样，肯定需要把形状调整，这么直接interpolate是最佳方案吗？
            resize_feats = F.interpolate( # 测试发现，interpolate会直接改变特征内的值，例如[1,2,3,4]-->inter to 8 size-->[1.0000, 1.2500, 1.7500, 2.2500, 2.7500, 3.2500, 3.7500, 4.0000]
                video_feats.unsqueeze(0),
                size=self.max_seq_len,
                mode='linear',
                align_corners=False
            )
            video_feats = resize_feats.squeeze(0)
            # print('interp video_feats:',video_feats.shape)
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
                # print('interp [audio_feats]:',audio_feats.shape)
          
            # feats=torch.cat([feats,audio_feats],dim=0)
        feats=torch.cat([video_feats,audio_feats],dim=0)
        
      
            # feats = feats / torch.max(feats) TODO 替换成正儿巴经从config读的
            # feats = torch.cat([audio_feats, audio_feats],dim=0) # TODO 假特征，为了直接用现有的ckpt，eval起来模型
            # feats = feats/400.
        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        
        if video_item['segments'] is not None:
            # feat_stride 的意义好像是将原始的frame_index映射到768长度下的新index
            # segments是[start_second, end_second]
            # print("segments origin:",video_item['segments'])
            # print(video_item['fps'])
            
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
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
                try:
                    segments = torch.stack(valid_seg_list, dim=0)
                except Exception as e:
                    # json_filename = os.path.join(json_rela_path)
                    print(json_rela_path)
                    print(segments)
                    print('duration:',video_item['duration'])
                    print(video_item['segments'])
                    print('fps:',video_item['fps'])
                    print('video feat shape:',np.load(os.path.join(self.video_feat_folder, feature_rela_path)).shape)
                    print('feat_stride:',feat_stride)
                    print(feat_offset)
                    raise RuntimeError
                # print("segments valid:",segments)
                labels = torch.cat(valid_label_list)
        else:
            segments, labels = None, None
            
        av_labels = torch.tensor(av_label_list)
        # print("!!! max[feats]:",torch.max(feats))
        # print("!! min(feats):",torch.min(feats))
        # return a data dict
        data_dict = {'video_id'        : json_rela_path.replace('.json','.mp4'),
                     'feats'           : feats,      # C x T 用到了
                     'segments'        : segments,   # N x 2 用到了 
                     'n_fakes'         : 0 if segments is None else segments.shape[0], # 为了eval新加的
                     'labels'          : labels,     # N 用到了
                     'av_labels'       : av_labels,
                     'fps'             : video_item['fps'],
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


@register_dataset("deepfake_video_audioBYOLA")
class DeepFakeVideoAudioDataset(Dataset):
    def __init__(
        self,
        # ,
        is_training,      # if in training mode
        split,            # split, a tuple/list allowing concat of subsets
        video_feat_folder,      # folder for features
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
        video_input_dim,        # input feat dim
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
        self.video_feat_folder = video_feat_folder
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
        self.video_input_dim = video_input_dim
        self.audio_input_dim=audio_input_dim
        self.default_fps = None
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = {'Fake':0}
        self.crop_ratio = crop_ratio
        # proposal vs action categories
        assert (num_classes == 1)
        self._get_train_list()#glob.glob(os.path.join('/home/ubuntu/shared-data/UMMAFormer/data/deepfake/train_metadata','*','*','*','real_video_fake_audio.json'))
        
        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'DeepFake_Audio',
            'tiou_thresholds': np.linspace(0.5, 0.95, 10), 
            'empty_label_ids': []
        }
        print("{} subset has {} videos".format(self.split,len(self.data_list)))

        self.byola_fps=12.497 # TODO 有些又是12.4几
        # self.emotion_fps = 50 

    def _get_train_list(self):
        data_list = []
        with open(self.train_txt,'r') as f:
            lines = f.readlines()
        for line in lines:
            data_list.append(line.strip())
        # random.shuffle(data_list)
        self.data_list = data_list
        print('data_list:',len(data_list))


    def get_attributes(self):
        return self.db_attributes

    

    def get_av_labels(self,json_filename):# 实际上返回的是video_audio_label
        if 'fake_video_real_audio.json' in json_filename:
            return [0,1]
        elif 'fake_video_fake_audio.json' in json_filename:
            return [0,0]
        elif 'real_video_fake_audio.json' in json_filename:
            return [1,0]
        elif 'real.json' in json_filename:
            return [1,1]
        else:
            raise ValueError(f"[Error] in get_av_labels() json file {json_filename} is incorrect")


    def _load_json_db(self, json_file):
        # load database and select the subset
        # print(f"================{json_file}===============")
        with open(os.path.join(self.json_folder,json_file), 'r') as fid:
            value = json.load(fid)


        # fill in the db (immutable afterwards)
        # dict_db = tuple()
        # for value in json_db:
        # key =  os.path.splitext(os.path.basename(value['file']))[0]
      
        
        
        duration = value['audio_frames'] / 16000
        # get fps if available
        if self.default_fps is not None:
            fps = self.default_fps
        elif 'fps' in value:
            fps = value['fps']
        elif 'video_frames' in value:
            fps = value['video_frames'] / duration # 这条分支
            # print('original fps:',fps)
        else:
            assert False, "Unknown video FPS."
        # duration = value['duration']
        # print('original fps:',fps)
        
       
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
        dict_db = {
                    'fps' : fps,
                    'duration' : duration,
                    'split': value['split'].lower(),
                    'segments' : segments,
                    'labels' : labels,
                        
                       
        }

        return dict_db

    def __len__(self):
        return len(self.data_list)

        # 用一个单独的文件列表存，再shuffle
        
    def error_item(self,idx):
        json_rela_path = self.data_list[idx//2]
        av_label_list = self.get_av_labels(json_rela_path)
        video_item = self._load_json_db(json_rela_path)
        feature_rela_path = json_rela_path.replace('.json','.npy')
        video_filename = os.path.join(self.video_feat_folder, feature_rela_path)
        video_feats = np.load(video_filename)
        audio_filename = os.path.join(self.audio_feat_folder, feature_rela_path)
        audio_feats = np.load(audio_filename)
        return av_label_list,video_feats,audio_feats

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        json_rela_path = self.data_list[idx]
        av_label_list = self.get_av_labels(json_rela_path)
        
        # print('json_file:',json_file)
        video_item = self._load_json_db(json_rela_path)

        feature_rela_path = json_rela_path.replace('.json','.npy')
        # load features
        if self.video_feat_folder is not None:
            video_filename = os.path.join(self.video_feat_folder, feature_rela_path)
            try:
                video_feats = np.load(video_filename)
                
            except ValueError as e:
                print(e)
                print(video_filename)
            # TODO 弄清楚这个需不需要截取
            # print("origin video_feats.shape:",video_feats.shape) # 需要shape[0]是时间 # (306,256)
            
        if self.audio_feat_folder is not None:
            audio_filename = os.path.join(self.audio_feat_folder, feature_rela_path)
            try:
                audio_feats = np.load(audio_filename)
                
            except ValueError as e:
                print(e)
                print(audio_filename)
                
               
            
            # 需要对这个进行截取
            audio_feats = audio_feats[:int(self.byola_fps*video_item['duration']-0.3657)]
            # print("resized audio_feats.shape:",audio_feats.shape) 
        

        # we support both fixed length features / variable length features
        # case 1: variable length features for training
        if self.feat_stride > 0 and (not self.force_upsampling): # 想走这一条
            # var length features
            feat_stride, num_frames = self.feat_stride, self.num_frames
            # only apply down sampling here
            if self.downsample_rate > 1:
                video_feats = video_feats[::self.downsample_rate, :]
                feat_stride = self.feat_stride * self.downsample_rate
        # case 2: variable length features for input, yet resized for training
        
        elif self.feat_stride > 0 and self.force_upsampling: # 这条分支
            feat_stride = float(
                (video_feats.shape[0] - 1) * self.feat_stride + self.num_frames
            ) / self.max_seq_len # 117/768 = 0.152 感觉像是当前特征的一行代表原始视频的多少FPS？ 不缩放的话默认是1FPS
            # center the features
            num_frames = feat_stride
        # case 3: fixed length features for input
        else:
            # deal with fixed length feature, recompute feat_stride, num_frames
            seq_len = video_feats.shape[0]
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

        
        video_feats = torch.from_numpy(np.ascontiguousarray(video_feats.transpose())) #现在变成了shape[-1]是时间

        # resize the features if needed
        if (video_feats.shape[-1] != self.max_seq_len) and self.force_upsampling: # 同样是问题1，但是涉及到多模态，或者是不同模型输出的特征做拼接时，不同特征的HZ不一样，肯定需要把形状调整，这么直接interpolate是最佳方案吗？
            resize_feats = F.interpolate( # 测试发现，interpolate会直接改变特征内的值，例如[1,2,3,4]-->inter to 8 size-->[1.0000, 1.2500, 1.7500, 2.2500, 2.7500, 3.2500, 3.7500, 4.0000]
                video_feats.unsqueeze(0),
                size=self.max_seq_len,
                mode='linear',
                align_corners=False
            )
            video_feats = resize_feats.squeeze(0)
            # print('interp video_feats:',video_feats.shape)
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
                # print('interp [audio_feats]:',audio_feats.shape)
          
            # feats=torch.cat([feats,audio_feats],dim=0)
        feats=torch.cat([video_feats,audio_feats],dim=0)
        
      
            # feats = feats / torch.max(feats) TODO 替换成正儿巴经从config读的
            # feats = torch.cat([audio_feats, audio_feats],dim=0) # TODO 假特征，为了直接用现有的ckpt，eval起来模型
            # feats = feats/400.
        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        
        if video_item['segments'] is not None:
            # feat_stride 的意义好像是将原始的frame_index映射到768长度下的新index
            # segments是[start_second, end_second]
            # print("segments origin:",video_item['segments'])
            # print(video_item['fps'])
            
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
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
                try:
                    segments = torch.stack(valid_seg_list, dim=0)
                except Exception as e:
                    # json_filename = os.path.join(json_rela_path)
                    print(json_rela_path)
                    print(segments)
                    print('duration:',video_item['duration'])
                    print(video_item['segments'])
                    print('fps:',video_item['fps'])
                    print('video feat shape:',np.load(os.path.join(self.video_feat_folder, feature_rela_path)).shape)
                    print('feat_stride:',feat_stride)
                    print(feat_offset)
                    raise RuntimeError
                # print("segments valid:",segments)
                labels = torch.cat(valid_label_list)
        else:
            segments, labels = None, None
            
        av_labels = torch.tensor(av_label_list)
        # print("!!! max[feats]:",torch.max(feats))
        # print("!! min(feats):",torch.min(feats))
        # return a data dict
        data_dict = {'video_id'        : json_rela_path.replace('.json','.mp4'),
                     'feats'           : feats,      # C x T 用到了
                     'segments'        : segments,   # N x 2 用到了 
                     'n_fakes'         : 0 if segments is None else segments.shape[0], # 为了eval新加的
                     'labels'          : labels,     # N 用到了
                     'av_labels'       : av_labels,
                     'fps'             : video_item['fps'],
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


@register_dataset("deepfake_video_audioEmoBYOLA")
class DeepFakeVideoAudioDataset(Dataset):
    def __init__(
        self,
        # ,
        is_training,      # if in training mode
        split,            # split, a tuple/list allowing concat of subsets
        video_feat_folder,      # folder for features
        audio_feat_folder,
        audio_byola_feat_folder, # folder for audio features
        audio_emo_feat_folder,
        train_txt,
        json_folder,        # json folder for annotations
        feat_stride,      # temporal stride of the feats
        num_frames,       # number of frames for each feat
        default_fps,      # default fps
        downsample_rate,  # downsample rate for feats 没看到这个
        max_seq_len,      # maximum sequence length during training
        trunc_thresh,     # threshold for truncate an action segment TODO 这是干啥的
        crop_ratio,       # a tuple (e.g., (0.9, 1.0)) for random cropping
        video_input_dim,        # input feat dim
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
        self.video_feat_folder = video_feat_folder
        self.audio_byola_feat_folder= audio_byola_feat_folder
        self.audio_emo_feat_folder= audio_emo_feat_folder
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
        self.video_input_dim = video_input_dim
        self.audio_input_dim=audio_input_dim
        self.default_fps = None
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = {'Fake':0}
        self.crop_ratio = crop_ratio
        # proposal vs action categories
        assert (num_classes == 1)
        self._get_train_list()#glob.glob(os.path.join('/home/ubuntu/shared-data/UMMAFormer/data/deepfake/train_metadata','*','*','*','real_video_fake_audio.json'))
        
        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'DeepFake_Audio',
            'tiou_thresholds': np.linspace(0.5, 0.95, 10), 
            'empty_label_ids': []
        }
        print("{} subset has {} videos".format(self.split,len(self.data_list)))

        self.byola_fps=12.497 
        self.emotion_fps = 50 

    def _get_train_list(self):
        data_list = []
        with open(self.train_txt,'r') as f:
            lines = f.readlines()
        for line in lines:
            data_list.append(line.strip())
        # random.shuffle(data_list)
        self.data_list = data_list
        print('data_list:',len(data_list))


    def get_attributes(self):
        return self.db_attributes

    

    def get_av_labels(self,json_filename):# 实际上返回的是video_audio_label
        if 'fake_video_real_audio.json' in json_filename:
            return [0,1]
        elif 'fake_video_fake_audio.json' in json_filename:
            return [0,0]
        elif 'real_video_fake_audio.json' in json_filename:
            return [1,0]
        elif 'real.json' in json_filename:
            return [1,1]
        else:
            # raise ValueError(f"[Error] in get_av_labels() json file {json_filename} is incorrect")
            return [-1,-1] # 因为现在把测试集也加进来了


    def _load_json_db(self, json_file):
        # load database and select the subset
        # print(f"================{json_file}===============")
        with open(os.path.join(self.json_folder,json_file), 'r') as fid:
            value = json.load(fid)


        # fill in the db (immutable afterwards)
        # dict_db = tuple()
        # for value in json_db:
        # key =  os.path.splitext(os.path.basename(value['file']))[0]
      
        
        
        duration = value['audio_frames'] / 16000
        # get fps if available
        if self.default_fps is not None:
            fps = self.default_fps
        elif 'fps' in value:
            fps = value['fps']
        elif 'video_frames' in value:
            fps = value['video_frames'] / duration # 这条分支
            # print('original fps:',fps)
        else:
            assert False, "Unknown video FPS."
        # duration = value['duration']
        # print('original fps:',fps)
        
       
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
        dict_db = {
                    'fps' : fps,
                    'duration' : duration,
                    'split': value['split'].lower(),
                    'segments' : segments,
                    'labels' : labels,
                        
                       
        }

        return dict_db

    def __len__(self):
        return len(self.data_list)

        # 用一个单独的文件列表存，再shuffle
        
   

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        json_rela_path = self.data_list[idx]
        av_label_list = self.get_av_labels(json_rela_path)
        
        # print('json_file:',json_file)
        video_item = self._load_json_db(json_rela_path)

        feature_rela_path = json_rela_path.replace('.json','.npy')
        # load features
        if self.video_feat_folder is not None:
            video_filename = os.path.join(self.video_feat_folder, feature_rela_path)
            try:
                video_feats = np.load(video_filename)
                
            except ValueError as e:
                print(e)
                print(video_filename)
            # TODO 弄清楚这个需不需要截取
            # print("origin video_feats.shape:",video_feats.shape) # 需要shape[0]是时间 # (306,256)
            
        if self.audio_byola_feat_folder is not None:
            audio_filename1 = os.path.join(self.audio_byola_feat_folder, feature_rela_path)
            try:
                audio_feats1 = np.load(audio_filename1)
                
            except ValueError as e:
                print(e)
                print(audio_filename1)
                
        if self.audio_emo_feat_folder is not None:
            audio_filename2 = os.path.join(self.audio_emo_feat_folder, feature_rela_path)
            try:
                audio_feats2 = np.load(audio_filename2)
                
            except ValueError as e:
                print(e)
                print(audio_filename2)       
            
            # 需要对这个进行截取
            audio_feats1 = audio_feats1[:int(self.byola_fps*video_item['duration']-0.3657)]
            audio_feats2 = audio_feats2[:int(self.emotion_fps*video_item['duration']-0.817)]
            # print("resized audio_feats.shape:",audio_feats.shape) 
        

        # we support both fixed length features / variable length features
        # case 1: variable length features for training
        if self.feat_stride > 0 and (not self.force_upsampling): # 想走这一条
            # var length features
            feat_stride, num_frames = self.feat_stride, self.num_frames
            # only apply down sampling here
            if self.downsample_rate > 1:
                video_feats = video_feats[::self.downsample_rate, :]
                feat_stride = self.feat_stride * self.downsample_rate
        # case 2: variable length features for input, yet resized for training
        
        elif self.feat_stride > 0 and self.force_upsampling: # 这条分支
            feat_stride = float(
                (video_feats.shape[0] - 1) * self.feat_stride + self.num_frames
            ) / self.max_seq_len # 117/768 = 0.152 感觉像是当前特征的一行代表原始视频的多少FPS？ 不缩放的话默认是1FPS
            # center the features
            num_frames = feat_stride
        # case 3: fixed length features for input
        else:
            # deal with fixed length feature, recompute feat_stride, num_frames
            seq_len = video_feats.shape[0]
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

        
        video_feats = torch.from_numpy(np.ascontiguousarray(video_feats.transpose())) #现在变成了shape[-1]是时间

        # resize the features if needed
        if (video_feats.shape[-1] != self.max_seq_len) and self.force_upsampling: # 同样是问题1，但是涉及到多模态，或者是不同模型输出的特征做拼接时，不同特征的HZ不一样，肯定需要把形状调整，这么直接interpolate是最佳方案吗？
            resize_feats = F.interpolate( # 测试发现，interpolate会直接改变特征内的值，例如[1,2,3,4]-->inter to 8 size-->[1.0000, 1.2500, 1.7500, 2.2500, 2.7500, 3.2500, 3.7500, 4.0000]
                video_feats.unsqueeze(0),
                size=self.max_seq_len,
                mode='linear',
                align_corners=False
            )
            video_feats = resize_feats.squeeze(0)
            # print('interp video_feats:',video_feats.shape)
        if (self.audio_byola_feat_folder is not None):
            audio_feats1 = torch.from_numpy(np.ascontiguousarray(audio_feats1.transpose()))
            if self.force_upsampling:
                resize_audio_feats1 = F.interpolate(
                    audio_feats1.unsqueeze(0),
                    size=self.max_seq_len,
                    mode='linear',
                    align_corners=False
                )
                audio_feats1 = resize_audio_feats1.squeeze(0)
                # print('interp [audio_feats]:',audio_feats.shape)
        if (self.audio_emo_feat_folder is not None):
            audio_feats2 = torch.from_numpy(np.ascontiguousarray(audio_feats2.transpose()))
            if self.force_upsampling:
                resize_audio_feats2 = F.interpolate(
                    audio_feats2.unsqueeze(0),
                    size=self.max_seq_len,
                    mode='linear',
                    align_corners=False
                )
                audio_feats2 = resize_audio_feats2.squeeze(0)
            # feats=torch.cat([feats,audio_feats],dim=0)
        feats=torch.cat([video_feats,audio_feats1,audio_feats2],dim=0)
        
      
            # feats = feats / torch.max(feats) TODO 替换成正儿巴经从config读的
            # feats = torch.cat([audio_feats, audio_feats],dim=0) # TODO 假特征，为了直接用现有的ckpt，eval起来模型
            # feats = feats/400.
        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        
        if video_item['segments'] is not None:
            # feat_stride 的意义好像是将原始的frame_index映射到768长度下的新index
            # segments是[start_second, end_second]
            # print("segments origin:",video_item['segments'])
            # print(video_item['fps'])
            
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
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
                try:
                    segments = torch.stack(valid_seg_list, dim=0)
                except Exception as e:
                    # json_filename = os.path.join(json_rela_path)
                    print(json_rela_path)
                    print(segments)
                    print('duration:',video_item['duration'])
                    print(video_item['segments'])
                    print('fps:',video_item['fps'])
                    print('video feat shape:',np.load(os.path.join(self.video_feat_folder, feature_rela_path)).shape)
                    print('feat_stride:',feat_stride)
                    print(feat_offset)
                    raise RuntimeError
                # print("segments valid:",segments)
                labels = torch.cat(valid_label_list)
        else:
            segments, labels = None, None
            
        av_labels = torch.tensor(av_label_list)
        # print("!!! max[feats]:",torch.max(feats))
        # print("!! min(feats):",torch.min(feats))
        # return a data dict
        data_dict = {'video_id'        : json_rela_path.replace('.json','.mp4'),
                     'feats'           : feats,      # C x T 用到了
                     'segments'        : segments,   # N x 2 用到了 
                     'n_fakes'         : 0 if segments is None else segments.shape[0], # 为了eval新加的
                     'labels'          : labels,     # N 用到了
                     'av_labels'       : av_labels,
                     'fps'             : video_item['fps'],
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

@register_dataset("deepfake_video_audioEmoBYOLA_THE")
class DeepFakeVideoAudioDataset(Dataset):
    def __init__(
        self,
        # ,
        is_training,      # if in training mode
        split,            # split, a tuple/list allowing concat of subsets
        video_feat_folder,      # folder for features
        audio_feat_folder,
        audio_byola_feat_folder, # folder for audio features
        audio_emo_feat_folder,
        train_txt,
        json_folder,        # json folder for annotations
        feat_stride,      # temporal stride of the feats
        num_frames,       # number of frames for each feat
        default_fps,      # default fps
        downsample_rate,  # downsample rate for feats 没看到这个
        max_seq_len,      # maximum sequence length during training
        trunc_thresh,     # threshold for truncate an action segment TODO 这是干啥的
        crop_ratio,       # a tuple (e.g., (0.9, 1.0)) for random cropping
        video_input_dim,        # input feat dim
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
        self.video_feat_folder = video_feat_folder
        self.audio_byola_feat_folder= audio_byola_feat_folder
        self.audio_emo_feat_folder= audio_emo_feat_folder
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
        self.video_input_dim = video_input_dim
        self.audio_input_dim=audio_input_dim
        self.default_fps = None
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = {'Fake':0}
        self.crop_ratio = crop_ratio
        # proposal vs action categories
        assert (num_classes == 1)
        self._get_train_list()#glob.glob(os.path.join('/home/ubuntu/shared-data/UMMAFormer/data/deepfake/train_metadata','*','*','*','real_video_fake_audio.json'))
        
        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'DeepFake_Audio',
            'tiou_thresholds': np.linspace(0.5, 0.95, 10), 
            'empty_label_ids': []
        }
        print("{} subset has {} videos".format(self.split,len(self.data_list)))

        self.byola_fps=12.497 
        self.emotion_fps = 50 

    def _get_train_list(self):
        data_list = []
        with open(self.train_txt,'r') as f:
            lines = f.readlines()
        for line in lines:
            data_list.append(line.strip())
        # random.shuffle(data_list)
        self.data_list = data_list
        print('data_list:',len(data_list))


    def get_attributes(self):
        return self.db_attributes
    

    def get_av_labels(self,json_filename):# 实际上返回的是video_audio_label
        if 'fake_video_real_audio.json' in json_filename:
            return [0,1]
        elif 'fake_video_fake_audio.json' in json_filename:
            return [0,0]
        elif 'real_video_fake_audio.json' in json_filename:
            return [1,0]
        elif 'real.json' in json_filename:
            return [1,1]
        else:
            # raise ValueError(f"[Error] in get_av_labels() json file {json_filename} is incorrect")
            return [-1,-1] 


    def _load_json_db(self, json_file):
        # load database and select the subset
        # print(f"================{json_file}===============")
        with open(os.path.join(self.json_folder,json_file), 'r') as fid:
            value = json.load(fid)


        # fill in the db (immutable afterwards)
        # dict_db = tuple()
        # for value in json_db:
        # key =  os.path.splitext(os.path.basename(value['file']))[0]
      
        
        
        duration = value['audio_frames'] / 16000
        # get fps if available
        if self.default_fps is not None:
            fps = self.default_fps
        elif 'fps' in value:
            fps = value['fps']
        elif 'video_frames' in value:
            fps = value['video_frames'] / duration # 这条分支
            # print('original fps:',fps)
        else:
            assert False, "Unknown video FPS."
        # duration = value['duration']
        # print('original fps:',fps)
        
       
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
        dict_db = {
                    'fps' : fps,
                    'duration' : duration,
                    'split': value['split'].lower(),
                    'segments' : segments,
                    'labels' : labels,
                        
                       
        }

        return dict_db

    def __len__(self):
        return len(self.data_list)

        # 用一个单独的文件列表存，再shuffle
        
    def map_segments_to_labels(self, data_dict):
        duration = data_dict['duration']
        segments = data_dict['segments']
        av_labels = data_dict['av_labels']
        labels = torch.zeros(768) # video,audio

        segment_length = duration / 768

        # 将视频和音频的假冒段映射到标签
        for i,segments in enumerate([segments]):
            for start, end in segments:
                # 计算假冒段在标签中的开始和结束位置
                start_idx = int(start / segment_length)
                end_idx = int(end / segment_length)

                # 将假冒段的标签设置为1
                if av_labels[0]>0 or av_labels[1]>0:
                    labels[start_idx:end_idx] = 1
                # if av_labels[1]>0:
                #     labels[1, start_idx:end_idx] = 1

        return labels

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        json_rela_path = self.data_list[idx]
        av_label_list = self.get_av_labels(json_rela_path)
        
        # print('json_file:',json_file)
        video_item = self._load_json_db(json_rela_path)

        feature_rela_path = json_rela_path.replace('.json','.npy')
        # load features
        if self.video_feat_folder is not None:
            video_filename = os.path.join(self.video_feat_folder, feature_rela_path)
            try:
                video_feats = np.load(video_filename)
                
            except ValueError as e:
                print(e)
                print(video_filename)
            # TODO 弄清楚这个需不需要截取
            # print("origin video_feats.shape:",video_feats.shape) # 需要shape[0]是时间 # (306,256)
            
        if self.audio_byola_feat_folder is not None:
            audio_filename1 = os.path.join(self.audio_byola_feat_folder, feature_rela_path)
            try:
                audio_feats1 = np.load(audio_filename1)
                
            except ValueError as e:
                print(e)
                print(audio_filename1)
                
        if self.audio_emo_feat_folder is not None:
            audio_filename2 = os.path.join(self.audio_emo_feat_folder, feature_rela_path)
            try:
                audio_feats2 = np.load(audio_filename2)
                
            except ValueError as e:
                print(e)
                print(audio_filename2)       
            
            # 需要对这个进行截取
            audio_feats1 = audio_feats1[:int(self.byola_fps*video_item['duration']-0.3657)]
            audio_feats2 = audio_feats2[:int(self.emotion_fps*video_item['duration']-0.817)]
            # print("resized audio_feats.shape:",audio_feats.shape) 
        

        # we support both fixed length features / variable length features
        # case 1: variable length features for training
        if self.feat_stride > 0 and (not self.force_upsampling): # 想走这一条
            # var length features
            feat_stride, num_frames = self.feat_stride, self.num_frames
            # only apply down sampling here
            if self.downsample_rate > 1:
                video_feats = video_feats[::self.downsample_rate, :]
                feat_stride = self.feat_stride * self.downsample_rate
        # case 2: variable length features for input, yet resized for training
        
        elif self.feat_stride > 0 and self.force_upsampling: # 这条分支
            feat_stride = float(
                (video_feats.shape[0] - 1) * self.feat_stride + self.num_frames
            ) / self.max_seq_len # 117/768 = 0.152 感觉像是当前特征的一行代表原始视频的多少FPS？ 不缩放的话默认是1FPS
            # center the features
            num_frames = feat_stride
        # case 3: fixed length features for input
        else:
            # deal with fixed length feature, recompute feat_stride, num_frames
            seq_len = video_feats.shape[0]
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

        
        video_feats = torch.from_numpy(np.ascontiguousarray(video_feats.transpose())) #现在变成了shape[-1]是时间

        # resize the features if needed
        if (video_feats.shape[-1] != self.max_seq_len) and self.force_upsampling: # 同样是问题1，但是涉及到多模态，或者是不同模型输出的特征做拼接时，不同特征的HZ不一样，肯定需要把形状调整，这么直接interpolate是最佳方案吗？
            resize_feats = F.interpolate( # 测试发现，interpolate会直接改变特征内的值，例如[1,2,3,4]-->inter to 8 size-->[1.0000, 1.2500, 1.7500, 2.2500, 2.7500, 3.2500, 3.7500, 4.0000]
                video_feats.unsqueeze(0),
                size=self.max_seq_len,
                mode='linear',
                align_corners=False
            )
            video_feats = resize_feats.squeeze(0)
            # print('interp video_feats:',video_feats.shape)
        if (self.audio_byola_feat_folder is not None):
            audio_feats1 = torch.from_numpy(np.ascontiguousarray(audio_feats1.transpose()))
            if self.force_upsampling:
                resize_audio_feats1 = F.interpolate(
                    audio_feats1.unsqueeze(0),
                    size=self.max_seq_len,
                    mode='linear',
                    align_corners=False
                )
                audio_feats1 = resize_audio_feats1.squeeze(0)
                # print('interp [audio_feats]:',audio_feats.shape)
        if (self.audio_emo_feat_folder is not None):
            audio_feats2 = torch.from_numpy(np.ascontiguousarray(audio_feats2.transpose()))
            if self.force_upsampling:
                resize_audio_feats2 = F.interpolate(
                    audio_feats2.unsqueeze(0),
                    size=self.max_seq_len,
                    mode='linear',
                    align_corners=False
                )
                audio_feats2 = resize_audio_feats2.squeeze(0)
            # feats=torch.cat([feats,audio_feats],dim=0)
        feats=torch.cat([video_feats,audio_feats1,audio_feats2],dim=0)
        
      
            # feats = feats / torch.max(feats) TODO 替换成正儿巴经从config读的
            # feats = torch.cat([audio_feats, audio_feats],dim=0) # TODO 假特征，为了直接用现有的ckpt，eval起来模型
            # feats = feats/400.
        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        
        if video_item['segments'] is not None:
            # feat_stride 的意义好像是将原始的frame_index映射到768长度下的新index
            # segments是[start_second, end_second]
            # print("segments origin:",video_item['segments'])
            # print(video_item['fps'])
            
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
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
                try:
                    segments = torch.stack(valid_seg_list, dim=0)
                except Exception as e:
                    # json_filename = os.path.join(json_rela_path)
                    print(json_rela_path)
                    print(segments)
                    print('duration:',video_item['duration'])
                    print(video_item['segments'])
                    print('fps:',video_item['fps'])
                    print('video feat shape:',np.load(os.path.join(self.video_feat_folder, feature_rela_path)).shape)
                    print('feat_stride:',feat_stride)
                    print(feat_offset)
                    raise RuntimeError
                # print("segments valid:",segments)
                labels = torch.cat(valid_label_list)
        else:
            segments, labels = None, None
            
        av_labels = torch.tensor(av_label_list)
        # print("!!! max[feats]:",torch.max(feats))
        # print("!! min(feats):",torch.min(feats))
        # return a data dict
        data_dict = {'video_id'        : json_rela_path.replace('.json','.mp4'),
                     'feats'           : feats,      # C x T 用到了
                     'segments'        : segments,   # N x 2 用到了 
                     'n_fakes'         : 0 if segments is None else segments.shape[0], # 为了eval新加的
                     'labels'          : labels,     # N 用到了
                     'av_labels'       : av_labels,
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'split'           : video_item['split'],
                     'feat_num_frames' : num_frames,
                     'gt_frame_labels' : None}

        # no truncation is needed
        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
            )
       
            gt_frame_labels = self.map_segments_to_labels(data_dict)
            data_dict['gt_frame_labels'] = gt_frame_labels
        
        return data_dict



# if __name__ == '__main__':
#     deepfake = DeepFakeVideoAudioDataset(is_training=True, split=['train'],
#     video_feat_folder='/home/ubuntu/sn15_share_dir/av_deepfake/train_frame_features',
#     audio_feat_folder='/home/ubuntu/sn15_share_dir/av_deepfake/train_wav',
#     train_txt = '/home/ubuntu/sn15_share_dir/av_deepfake/preprocess_v2/train_video_audio.txt',
#     json_folder = '/home/ubuntu/sn15_share_dir/av_deepfake/train_metadata',
#     feat_stride=1,
#     num_frames=1,
#     default_fps=None,
#     downsample_rate=0,max_seq_len=768,trunc_thresh=0.5,
#     crop_ratio=[0.9,1.0],video_input_dim=256,audio_input_dim=768,
#     num_classes=1,file_prefix='rgb',file_ext='.npy',audio_file_ext=None,force_upsampling=True)
#     data = deepfake[1]
#     print(data['video_id'])
#     print(data['duration'])
#     print(data['feats'].shape)
#     print(data['segments'])
#     print(data['labels']) #fake=0,real=1
#     print(data['av_labels'])# [1,1]
#     print(data['fps'])
