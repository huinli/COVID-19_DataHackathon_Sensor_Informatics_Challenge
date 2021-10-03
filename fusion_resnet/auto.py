import os
import shutil

RAW_DATASET_PATH = "data"

label_covid = {}
label_gender = {}
with open(RAW_DATASET_PATH+"/metadata.csv") as f:
    next(f)
    for line in f:
        line = line.replace("\n", "")
        splits = line.split(" ")
        label_covid[splits[0]] = 0 if splits[1] == "n" else 1
        label_gender[splits[0]] = splits[2]


def do(audio_type, list_id):

    LIST_PATH = RAW_DATASET_PATH+"/LISTS"
    AUDIO_PATH = RAW_DATASET_PATH+"/AUDIO_WAV/"+audio_type
    target_data_path = "repos/auto/data"
    
    target_data_path = target_data_path + "/" + audio_type + "_" + str(list_id)
    train = []
    validate = []

    with open(LIST_PATH+"/train_{}.csv".format(list_id)) as f:
        for line in f:
            line = line.replace("\n", "")
            train.append(line)
    with open(LIST_PATH+"/val_{}.csv".format(list_id)) as f:
        for line in f:
            line = line.replace("\n", "")
            validate.append(line)

    if os.path.exists(target_data_path):
        os.system("rm -rf "+target_data_path)

    os.mkdir(target_data_path)
    os.mkdir(target_data_path+"/dev")
    os.mkdir(target_data_path+"/test")
    os.mkdir(target_data_path+"/dev/wav")
    os.mkdir(target_data_path+"/test/wav")
    os.mkdir(target_data_path+"/dev/wav/id0")
    os.mkdir(target_data_path+"/dev/wav/id1")
    os.mkdir(target_data_path+"/test/wav/id0")
    os.mkdir(target_data_path+"/test/wav/id1")
    for file in train:
        os.mkdir(target_data_path +
                 "/dev/wav/id{}/{}/".format(label_covid[file], file))
        shutil.copy(AUDIO_PATH+"/"+file+".wav", target_data_path +
                    "/dev/wav/id{}/{}/{}.wav".format(label_covid[file], file, file))
    for file in validate:
        os.mkdir(target_data_path +
                 "/test/wav/id{}/{}/".format(label_covid[file], file))
        shutil.copy(AUDIO_PATH+"/"+file+".wav", target_data_path +
                    "/test/wav/id{}/{}/{}.wav".format(label_covid[file], file, file))

    with open(target_data_path+"/iden_split.txt", "a")as out:
        for file in train:
            out.write("{} id{}/{}/{}.wav\n".format(1,
                                                   label_covid[file], file, file))
        for file in validate:
            out.write("{} id{}/{}/{}.wav\n".format(2,
                                                   label_covid[file], file, file))
            out.write("{} id{}/{}/{}.wav\n".format(3,
                                                   label_covid[file], file, file))
    with open(target_data_path+"/veri_test.txt", "a")as out:
        out.write("")

    os.chdir("repos/auto")
    os.system("python data_preprocess.py data")
    os.system(
        "python train_baseline_identification.py --cfg exps/baseline/resnet34_iden.yaml")
    # os.system("python train_baseline_identification.py --cfg exps/baseline/resnet18_iden.yaml")

    # os.system("python test.py {}_{}_log/resnet18".format(audio_type,list_id))
    # os.system("python test.py {}_{}_log/resnet34".format(audio_type,list_id))

    if not os.path.exists("saved_logs"):
        os.mkdir("saved_logs")
    if not os.path.exists("saved_data"):
        os.mkdir("saved_data")

    # rename tmp logs
    log_path = "logs"
    for f in os.listdir(log_path):
        if "resnet18" in f:
            os.system("mv "+log_path+"/"+f+" "+log_path+"/resnet18")
        if "resnet34" in f:
            os.system("mv "+log_path+"/"+f+" "+log_path+"/resnet34")

    os.system("mv logs saved_logs/{}_{}_log".format(audio_type, list_id))
    os.system("mv data saved_data/{}_{}_data".format(audio_type, list_id))
    os.chdir("../..")

types = ["breathing", "cough", "speech"]
if __name__ == "__main__":
    
    for i in range(5):
        for t in types:
            if i==0 and t=="breathing":
                continue
            do(t, i)
