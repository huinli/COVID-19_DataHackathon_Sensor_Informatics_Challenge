import os
logF = "/logs/"


def log(x, end='\n'):
    print(x, end=end)
    with open('summary.txt', 'a') as f:
        f.write(x+end)


savesfs = os.listdir('saves')
savesfs.sort()
for save in savesfs:
    if os.path.exists('saves/'+save+logF):
        for model in os.listdir('saves/'+save + logF):
            for file in os.listdir('saves/'+save+logF + model+"/Log/"):
                if 'log' in file:
                    with open('saves/'+save+logF + model+"/Log/"+file) as f:
                        max_acc = 0
                        for line in f:
                            if 'Test Acc@' in line:
                                acc = float(line.split(' ')[4])
                                if max_acc < acc:
                                    max_acc = acc
                    log(save+'-'*(12-len(save)) + "\t" + model.split('_')[0]+"---"
                        "acc:" + str(max_acc)+"%")
        log('')
