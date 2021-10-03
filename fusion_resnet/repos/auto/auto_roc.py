
import os 



types = ["breathing","cough","speech"]
# types = ["breathing"]
if __name__ == "__main__":

    for i in range(5):
        for t in types:
            if(os.path.exists("data")):
                os.system("rm -rf data")
            os.system("cp -r saved_data/{}_{}_data data".format(t,i))
            os.system("python test.py saved_logs/{}_{}_log/resnet18".format(t,i))
            os.system("python test.py saved_logs/{}_{}_log/resnet34".format(t,i))
            