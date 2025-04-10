import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import glob
from typing import Optional, List, Callable, Any, Union, Tuple
import numpy as np
import einops
import torch
import torchaudio 
import torchvision
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F, Identity
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# import toml
from model import AlignVideo 
from tqdm import tqdm

def read_video(path: str):
    video, _, _ = torchvision.io.read_video(path, pts_unit="sec")
    video = video.permute(0, 3, 1, 2) / 255
    
    return video

def padding_video(tensor: Tensor, target: int, padding_method: str = "zero", padding_position: str = "tail") -> Tensor:
    t, c, h, w = tensor.shape
    padding_size = target - t

    # pad = _get_padding_pair(padding_size, padding_position)
    pad = [0, padding_size] # tail

    if padding_method == "zero":
        return F.pad(tensor, pad=[0, 0, 0, 0, 0, 0] + pad)
   

def resize_video(tensor: Tensor, size: Tuple[int, int], resize_method: str = "nearest") -> Tensor:
    return F.interpolate(tensor, size=size, mode=resize_method)


class VideoData(Dataset):

    def __init__(self, data_list: str, root: str = "data", frame_padding: int = 512,
        max_duration: int = 40, fps: int = 25,
        video_transform: Callable[[Tensor], Tensor] = Identity(),
        
    ):
        self.data_list = data_list # 需要提取特征的视频list
       
        self.video_padding = frame_padding
       
        self.max_duration = max_duration

    def __getitem__(self, index: int) -> List[Tensor]:
        file = self.data_list[index]
        video = read_video(file)
        video_len = video.shape[0]
        # print('video origin shape:',video.shape) # ([294, 3, 224, 224]
        if video_len<self.video_padding:
            video = padding_video(video, target=self.video_padding)
            # print('video small shape:',video.shape)
            video = rearrange(resize_video(video, (96, 96)), "t c h w -> c t h w")
            video = video.view(1,video.shape[0],video.shape[1],video.shape[2],video.shape[2])
            # print('video review shape:',video.shape)
            return video,video_len,file
        else:
            # 将帧数组切分为大小为512的块
            stack_video = [] 
            for i in range(video_len//self.video_padding + 1):
                video_sub = video[i*self.video_padding:(i+1)*self.video_padding]
                if video_sub.shape[0]<self.video_padding:
                    video_sub = padding_video(video_sub, target=self.video_padding)
                video_sub = rearrange(resize_video(video_sub, (96, 96)), "t c h w -> c t h w")
                stack_video.append(video_sub)
            stack_video = torch.stack(stack_video)
            # print('video large shape:',stack_video.shape)
             
            return stack_video,video_len,file


    def __len__(self) -> int:
        return len(self.data_list)

def abs_to_relative_path(path):
    items = path.split(os.path.sep)
    return os.path.join(items[-4],items[-3],items[-2],items[-1].replace('.mp4','.npy'))

def collate_fn(batch):
    videos, video_lens, files = zip(*batch)

    # 将所有的视频块连接在一起
    videos = [video for video_list in videos for video in video_list]

    # 将video_lens和files转换为列表
    video_lens = list(video_lens)
    files = list(files)

    return videos, video_lens,files

tmp_train_list = []
with open("/home/ubuntu/sn15_share_dir/av_deepfake/preprocess_v3/previous_not_ok_videos.txt","r") as fid:
    lines = fid.readlines()
for line in tqdm(lines):
    # file, duration = line.split(',')
    tmp = line.strip().replace('.json','.mp4')
    train_path = os.path.join('/home/ubuntu/sn15_share_dir/av_deepfake/train',tmp)
    val_path = os.path.join('/home/ubuntu/sn15_share_dir/av_deepfake/val/val_mp4',tmp)
    if not os.path.exists(train_path):
        if os.path.exists(val_path):
            tmp_train_list.append(val_path)
        else:
            print("error: no mp4",tmp)
    else:
        tmp_train_list.append(train_path)



print("total train list:",len(tmp_train_list))
BASE = '/home/ubuntu/sn15_share_dir/av_deepfake/train_frame_features/aligned_video'


device = 'cuda:0'
checkpoint = 'alignvideo_model.ckpt'
model = AlignVideo.load_from_checkpoint(checkpoint).to(device)
model.eval()


training_data = VideoData(data_list=tmp_train_list)
train_dataloader = DataLoader(training_data, batch_size=10, shuffle=False,collate_fn=collate_fn)
for i,(data,video_len,file) in enumerate(tqdm(train_dataloader)):
    
    data = torch.stack(data)
   
    with torch.no_grad():
        v_features = model.forward_features(data.to(device))
   
    v_features = v_features.permute(0,2,1)
    
   
    index = 0
    for video_i, cur_len in enumerate(video_len):
        # print(f'==[video:{video_i}], len:{cur_len}==')
        chunks_num = cur_len//512 + 1 
       
        tmp_feature = []
        for chunk_i in range(chunks_num-1):
            # print(f'append {index+chunk_i} 512 dim ')
            tmp_feature.append(v_features[index+chunk_i,:,:])#[n,512,256]
        # print(f'append {index+chunks_num -1} {cur_len%512} dim ')
        tmp_feature.append(v_features[index+chunks_num-1,:cur_len%512])
        cur_large_feature = torch.concat(tmp_feature,dim=0) 
        
        
        # print('cur_large_feature:',cur_large_feature.shape)
        index += chunks_num
        file_one = file[video_i]


   
        rela_save_path = abs_to_relative_path(file_one)
        save_path = os.path.join(BASE,rela_save_path)

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        np.save(save_path,cur_large_feature.cpu().numpy())
    
