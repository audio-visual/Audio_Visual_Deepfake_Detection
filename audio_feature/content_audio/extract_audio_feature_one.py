import argparse
from byol_a.common import *
from byol_a.augmentations import PrecomputedNorm
from byol_a.models import AudioNTT2020,AudioNTT2020Task6
from tqdm import tqdm 
import glob 
def get_test_file_list():
    results = []
    base = '/home/ubuntu/sn15_share_dir/av_deepfake/test/test_wav'
    

    for file in  glob.glob(os.path.join(base,'*.wav')):
        
        results.append(file)
        
    print("total wav files:",len(results))                       
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--csv_file', type=str, default=None, help='CSV file name')
    parser.add_argument('--device_id', type=int, default=0, help='GPU device ID')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device_id}')
    cfg = load_yaml_config('config.yaml')
    print(cfg)

    stats = [-2.2800865,  3.5897882]

    # Preprocessor and normalizer.
    to_melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
    )
    normalizer = PrecomputedNorm(stats)

    model = AudioNTT2020Task6(d=cfg.feature_d,n_mels=64)
    model.load_weight('pretrained_weights/AudioNTT2020-BYOLA-64x96d2048.pth', device)
    model = model.to(device)  # Move the model to the correct device
    model.eval()
    if args.csv_file is None:
        file_list = get_test_file_list() 
        
    else:
        with open(args.csv_file, 'r') as f:
            file_list = f.readlines()
    # Convert to a log-mel spectrogram, then normalize.
    for file in tqdm(file_list):
       
        # print(file)
        # print(os.path.exists(file.strip())
        # file=file.strip().replace('val_metadata','val/val_wav').replace('.json','.wav')
        # batch_npy = np.load(file.replace('.wav','.npy')) #(T,2024)
        # # print(batch_npy.shape)
        wav,sr=torchaudio.load(file.strip())
        # print('wav duration:',wav.size(1)/sr)
        assert sr==cfg.sample_rate
        try:
            lms = normalizer((to_melspec(wav) + torch.finfo(torch.float).eps).log())
        except Exception as e:
            print(file)
            print(e)
        # lms_reshape = torch.zeros(1,64,2541)
        # lms_reshape[0,:,:lms.size(-1)] = lms 
        # print('oringal:',lms.size())  
        # print('reshape:',lms_reshape.size())  
        with torch.no_grad():
            features = model(lms.unsqueeze(0).to(device))  # Ensure the input data is on the correct device

        features= features.squeeze(0).cpu().numpy()
        # print(features.shape)
        save_name = file.replace('test_wav','test_byola').replace('.wav','.npy')
        # dirs = os.path.dirname(save_name)
        # if not os.path.exists(dirs):
        #     os.makedirs(dirs,exist_ok=True)
        np.save(save_name,features)
        # print(save_name)
        # check if features numpy is close enough to batch_npy
        # print(np.allclose(features,batch_npy))
        # print(features[:10,:10])
        # print(batch_npy[:10,:10])
        # if i>10:
        #     break