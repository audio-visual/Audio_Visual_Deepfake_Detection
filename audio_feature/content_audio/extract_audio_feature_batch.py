import argparse
from byol_a.common import *
from byol_a.augmentations import PrecomputedNorm
from byol_a.models import AudioNTT2020,AudioNTT2020Task6

import torchaudio
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
MP4_train_base = '/home/ubuntu/sn15_share_dir/av_deepfake/train'

def get_filepath(txt_rela_file):
    tmp_train_list = []
    with open(txt_rela_file,"r") as fid:
        for line in fid:
            # file, duration = line.split(',')
            tmp = line.strip().replace('.npy','.wav')
            train_path = os.path.join('/home/ubuntu/sn15_share_dir/av_deepfake/train_wav_new',tmp)
     
            if not os.path.exists(train_path):
                
                train_mp4_file = os.path.join(MP4_train_base,tmp.replace('.wav','.mp4'))
                val_mp4_file = os.path.join(MP4_val_base,tmp.replace('.wav','.mp4'))
                exist_path = train_mp4_file if os.path.exists(train_mp4_file) else val_mp4_file
                cmd = ['ffmpeg', '-nostats', '-hide_banner', '-i', exist_path, '-map', '0:a', '-y', train_path]
                # subprocess.run(' '.join(cmd))
                # print(' '.join(cmd))
                os.system(' '.join(cmd))
                tmp_train_list.append(train_path)
            else:
                tmp_train_list.append(train_path)
            print(len(tmp_train_list))
           
    return tmp_train_list
# write a dataloader that can read and load wav files ,transform to melspectrom forming a batch,the time is set to the max length of the batch
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, txt_file, cfg, stats=None):
        # self.data = pd.read_csv(csv_file)
        self.data = get_filepath(txt_file)
        self.to_melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
        )
        self.normalizer = PrecomputedNorm(stats)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # wav, sr = torchaudio.load(self.data.iloc[idx, 0])
        wav,sr = torchaudio.load(self.data[idx])
        assert sr == cfg.sample_rate
        lms = self.normalizer((self.to_melspec(wav) + torch.finfo(torch.float).eps).log())
        # print('lms size:',lms.size())
        item = {'lms': lms,'path':self.data[idx]}
        return item
    
    def collate_fn(batch):
        max_len = max([item['lms'].size(2) for item in batch])
        lms_batch = torch.zeros(len(batch),1, batch[0]['lms'].size(1), max_len)
        path_batch = []
        data = {}
        for i, item in enumerate(batch):
            lms = item['lms']
            path = item['path']
            # if path =='/media/cwy/8T/deepfake/train/train_wav/id07639/xivWrEiCeI4/00041/real.wav':
                # print("="*10)
                # print(lms_batch.shape)
                # print("="*10)
            path_batch.append(path)
            lms_batch[i, :,:, :lms.size(2)] = lms
        data['lms'] = lms_batch
        data['path'] = path_batch
        return data
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_file', type=str, default='/home/ubuntu/sn15_share_dir/av_deepfake/preprocess_v3/val_byola_todo2.txt', help='CSV file name')
    parser.add_argument('--device_id', type=int,  default=1, help='GPU device ID')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size')

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device_id}')
    cfg = load_yaml_config('config.yaml')
    print(cfg)

    stats = [-2.2800865,  3.5897882]

    dataloader = DataLoader(AudioDataset(args.txt_file, cfg, stats), batch_size=args.batch_size, shuffle=False, collate_fn=AudioDataset.collate_fn)

    model = AudioNTT2020Task6(d=cfg.feature_d,n_mels=64)
    model.load_weight('pretrained_weights/AudioNTT2020-BYOLA-64x96d2048.pth', device)
    model = model.to(device)  # Move the model to the correct device

    SAVE_BASE = '/home/ubuntu/sn15_share_dir/av_deepfake/train_wav_new'
    for data in tqdm(dataloader):
        lms = data['lms']
        paths = data['path']
        # print(lms.size())
        with torch.no_grad():
            features = model(lms.to(device))  # Ensure the input data is on the correct device

        features= features.cpu().numpy()
        for i,file in enumerate(paths):
            
            # print(save_name)
            if 'val/val_wav' in file:
                save_name = file.replace('val/val_wav','train_wav_new').replace('.wav','.npy')
                
            else:
                save_name = os.path.splitext(file.strip())[0] + '.npy'
            if not os.path.exists(os.path.dirname(save_name)):
                    os.makedirs(os.path.dirname(save_name))
            # print(save_name)
            np.save(save_name,features[i])
        # print(save_name)