from pathlib import PurePath
from pydub import AudioSegment
import argparse
from os import listdir
from os import path
from os import mkdir
import shutil

def convert(path, save_path):
    flac_tmp_audio_data = AudioSegment.from_file(path)
    print(save_path)
    flac_tmp_audio_data.export(save_path, format="wav")

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', metavar='N', 
                    default="./AUDIO")
args = parser.parse_args()

audio_path = args.path
wav_path = audio_path + "_wav"
if(path.exists(wav_path)):
    shutil.rmtree(wav_path)
mkdir(wav_path)

folders = [f for f in listdir(audio_path)]
for folder in folders:
    print(folder)
    folder_full_path = path.join(audio_path, folder)
    mkdir(path.join(wav_path, folder))
    files = [f for f in listdir(folder_full_path)]
    for audio_file in files:
        full_audio_path = path.join(audio_path, folder, audio_file)
        save_audio_path = path.join(wav_path, folder, path.splitext(audio_file)[0]+ ".wav")
        convert(full_audio_path, save_audio_path)
