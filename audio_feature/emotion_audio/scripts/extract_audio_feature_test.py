import argparse
from dataclasses import dataclass
import numpy as np
import soundfile as sf
import pandas as pd
import torch
import torch.nn.functional as F
import fairseq

import torchaudio
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, text_file, stats=None):
        self.data = pd.read_csv(text_file)
    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        wav, sr = sf.read(self.data.iloc[idx, 0])
        # channel = sf.info(self.data.iloc[idx, 0]).channels
        # assert sr == 16e3, "Sample rate should be 16kHz, but got {}in file {}".format(sr, self.data.iloc[idx, 0])
        # assert channel == 1, "Channel should be 1, but got {} in file {}".format(channel, self.data.iloc[idx, 0])


        item = {'wav': wav,'path':self.data.iloc[idx, 0]}
        return item
    
    def collate_fn(batch):
        max_len = max([item['wav'].size for item in batch])
        # print(max_len)
        lms_batch = torch.zeros(len(batch), max_len)
        path_batch = []
        len_batch = []
        data = {}
        for i, item in enumerate(batch):
            wav = item['wav']
            path = item['path']
            path_batch.append(path)
            len_batch.append(wav.size/16000)
            lms_batch[i, :wav.size] = torch.FloatTensor(wav)
        data['wav'] = lms_batch
        data['path'] = path_batch
        data['len'] = len_batch
        return data

@dataclass
class UserDirModule:
    user_dir: str

def main():

    device=torch.device('cuda:0')


    model_dir = '../upstream'
    checkpoint_dir = 'pretrained/emotion2vec_base.pt'
    granularity = 'frame'
    # text_file='processed_byola_modified.txt'
    text_file='split_test01_left.txt' 


    dataloader = DataLoader(AudioDataset(text_file,), batch_size=15, shuffle=False, num_workers=4, collate_fn=AudioDataset.collate_fn)

    model_path = UserDirModule(model_dir)
    fairseq.utils.import_user_module(model_path)
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_dir])
    model = model[0]
    model.eval()
    model.to(device)


    for data in tqdm(dataloader):
        wav=data['wav']
        paths=data['path']
        lens = data['len']
        with torch.no_grad():
            feats=model.extract_features(wav.to(device),padding_mask=None)            
            feats = feats['x'].squeeze(0).cpu().numpy()
            # print(feats.shape) #(bs, time,768)

            for i,file in enumerate(paths):
                file = paths[i]
                feat = feats[i] 
                feat = feat[:int(50*lens[i]-0.817),:]
                # audio_feats[:int(self.emotion_fps*video_item['duration']-0.817)]
                save_name=os.path.splitext(file.strip())[0] + '.npy'
                save_name=save_name.replace('test_wav','test_wav_emotion2vec')
                # save_name=save_name.replace('train_wav_new','train_wav')
                # print(file)
                # print(feat.shape)
                np.save(save_name, feat)




    # if source_file.endswith('.wav'):
    #     wav, sr = sf.read(source_file)
    #     channel = sf.info(source_file).channels
    #     assert sr == 16e3, "Sample rate should be 16kHz, but got {}in file {}".format(sr, source_file)
    #     assert channel == 1, "Channel should be 1, but got {} in file {}".format(channel, source_file)
        
    
    # with torch.no_grad():
    #     source = torch.from_numpy(wav).float().cuda()
    #     if task.cfg.normalize:
    #         source = F.layer_norm(source, source.shape)
    #     source = source.view(1, -1)
    #     try:
    #         feats = model.extract_features(source, padding_mask=None)
    #         feats = feats['x'].squeeze(0).cpu().numpy()
    #         if granularity == 'frame':
    #             feats = feats
    #         elif granularity == 'utterance':
    #             feats = np.mean(feats, axis=0)
    #         else:
    #             raise ValueError("Unknown granularity: {}".format(args.granularity))
    #         print(feats.size())
    #         np.save(target_file, feats)
    #     except:
    #         Exception("Error in extracting features from {}".format(source_file))
    

if __name__ == '__main__':
    main()