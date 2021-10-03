import os
import shutil
import sys

# python GenWAV.py IEEE_HealthCareSummit_Dev_Data_Release

# RAW_DATASET_PATH = "IEEE_HealthCareSummit_Dev_Data_Release"
RAW_DATASET_PATH = sys.argv[1]

os.mkdir(RAW_DATASET_PATH+"/AUDIO_WAV")
for folder in ["breathing", "cough", "speech"]:
    os.mkdir(RAW_DATASET_PATH+"/AUDIO_WAV/"+folder)
    for file in os.listdir(RAW_DATASET_PATH+"/AUDIO/"+folder):
        os.system("ffmpeg -i {} {}".format(
            RAW_DATASET_PATH+"/AUDIO/"+folder+"/"+file,
            RAW_DATASET_PATH+"/AUDIO_WAV/"+folder+"/"+file.replace("flac", "wav"))
        )
