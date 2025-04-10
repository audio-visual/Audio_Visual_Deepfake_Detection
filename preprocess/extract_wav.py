import glob 
import os 
import time 
from multiprocessing import Pool
import subprocess

def extract_audio_wav(line):
    """Extract the audio wave from video streams using FFMPEG."""
    # output_file = os.path.join(line.replace('train_wav','train_wav_new'))[:-1]
    output_file = line
    input_line = os.path.join(line.replace('.wav','.mp4').replace('train_wav_new','train'))
    
    os.makedirs(os.path.dirname(output_file),exist_ok=True)
    try:
        if os.path.exists(output_file):
            return
        # cmd = ['ffmpeg', '-loglevel', 'quiet', '-nostats', '-hide_banner', '-i', input_line, '-map', '0:a', '-preset', 'ultrafast', '-y', output_file]
        cmd = ['ffmpeg', '-nostats', '-hide_banner', '-i', input_line, '-map', '0:a', '-preset', 'ultrafast', '-y', output_file]
        # subprocess.run(' '.join(cmd))
        print(' '.join(cmd))
        os.system(' '.join(cmd))
        
    except BaseException as e:
        print(e)
        with open('extract_wav_err_file_new1.txt', 'a+') as f:
            f.write(f'{line}\n')



if __name__ == '__main__':

   
    files_need_to_process = []
    train_mp4_files = glob.glob('/home/ubuntu/sn15_share_dir/av_deepfake/train/*/*/*/*.mp4')
    for train_mp4_file in train_mp4_files:
        train_wav_file = train_mp4_file.replace('.mp4','.wav').replace('train','train_wav_new')
        if not os.path.exists(train_wav_file):
            files_need_to_process.append(train_wav_file)

    print(f'left {len(files_need_to_process)} to extract')


    with Pool(16) as p:
        p.map(extract_audio_wav, files_need_to_process)