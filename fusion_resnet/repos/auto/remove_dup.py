import os
num = 0

the_path1 = "./data/breathing_" + str(num)
the_path2 = "./data/cough_" + str(num)
the_path3 = "./data/speech_" + str(num)

#dev_path1 = "/feature/dev/id1/"
#dev_path1 = "/feature/test/id0/"
dev_path1 = "/feature/merged/id0/

list1 = the_path1 + dev_path1

list11 = []
for files in os.listdir(list1):
    if '.npy' in files:
        list11.append(files)
print(len(list11))

list2 = the_path2 + dev_path1

list21 = []
for files in os.listdir(list2):
    if '.npy' in files:
        list21.append(files)
print(len(list21))

list3 = the_path3 + dev_path1

list31 = []
for files in os.listdir(list3):
    if '.npy' in files:
        list31.append(files)
print(len(list31))

common = [x for x in list11 if (x in list21 and x in list31)]
notin1 = [x for x in list11 if x not in common]
notin2 = [x for x in list21 if x not in common]
notin3 = [x for x in list31 if x not in common]

for ff in notin1:
    os.remove(list1 + ff)

for ff in notin2:
    os.remove(list2 + ff)

for ff in notin3:
    os.remove(list3 + ff)
