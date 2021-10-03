## Environmet

### requirements: Conda, FFmpeg

```
conda create -n xy_resnet python=3.7
conda activate xy_resnet
pip install -r requirements.txt
```

## Data Preprocess
To generate WAV audios from original dataset.
```
python GenWAV.py <DATASET_PATH>
```
for example:

```
python GenWAV.py ../IEEE_HealthCareSummit_Dev_Data_Release
```

## Run
```
conda activate xy_resnet
python auto.py
```
Then the code will run all 3 audio types and 5 list.
you can change the last part of `auto.py` to run specific setting.

## Model
The code is from AutoSpeech.
see https://github.com/VITA-Group/AutoSpeech

## Notes.
For each run, it requires ~5G disk space. run all the settings requires ~100GB.

The code is develop and run on linux. For windows, you may need to check and change commands like `"rm -rf"` in auto.py.

## Run Fusion Model

After you run the auto.py, and get the extracted feature, you should have all extracted feature under folder repos/data/. follow the format <<name_(number of set)_data>>. 

You should also get the best model for all three processes (cough, speech and breath) and save them under folder named ''checkpoint''.

Run
```
python train_FFN.py --cfg exps/baseline/resnet34_iden.yaml --load_path ./ --fold_num 1


python test_FFN.py --cfg exps/baseline/resnet34_iden.yaml --load_path ./ --fold_num 2
```


