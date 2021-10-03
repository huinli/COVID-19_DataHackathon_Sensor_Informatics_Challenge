from __future__ import print_function


import numpy as np
import torch.utils.data as data
from data_objects.speaker import Speaker
from torchvision import transforms as T
from data_objects.transforms import Normalize, TimeReverse, generate_test_sequence


def find_classes(speakers):
    classes = list(set([speaker.name for speaker in speakers]))
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


class DeepSpeakerDataset(data.Dataset):

    def __init__(self, data_dir1, data_dir2, data_dir3,sub_dir, partial_n_frames, partition=None, is_test=False):
        super(DeepSpeakerDataset, self).__init__()
        self.data_dir1 = data_dir1
        self.data_dir2 = data_dir2
        self.data_dir3 = data_dir3
        self.root1 = data_dir1.joinpath('feature', sub_dir)
        self.root2 = data_dir2.joinpath('feature', sub_dir)
        self.root3 = data_dir3.joinpath('feature', sub_dir)

        self.partition = partition
        self.partial_n_frames = partial_n_frames
        self.is_test = is_test

        speaker_dirs1 = [f for f in self.root1.glob("*") if f.is_dir()]
        speaker_dirs2 = [f for f in self.root2.glob("*") if f.is_dir()]
        speaker_dirs3 = [f for f in self.root3.glob("*") if f.is_dir()]

        if len(speaker_dirs1) == 0 or len(speaker_dirs2) == 0 or len(speaker_dirs3) == 0:
            raise Exception("No speakers found. Make sure you are pointing to the directory "
                            "containing all preprocessed speaker directories.") 
        
        self.speakers1 = [Speaker(speaker_dir, self.partition)
                         for speaker_dir in speaker_dirs1]

        self.speakers2 = [Speaker(speaker_dir, self.partition)
                         for speaker_dir in speaker_dirs2]

        self.speakers3 = [Speaker(speaker_dir, self.partition)
                         for speaker_dir in speaker_dirs3]
	
        print(len(self.speakers1))
        print(len(self.speakers2))
        print(len(self.speakers3))
        classes1, class_to_idx1 = find_classes(self.speakers1)
        classes2, class_to_idx2 = find_classes(self.speakers2)
        classes3, class_to_idx3 = find_classes(self.speakers3)

        self.classes1 = classes1
        self.classes2 = classes2
        self.classes3 = classes3

        sources1 = []
        for speaker in self.speakers1:
            sources1.extend(speaker.sources)

        sources2 = []
        for speaker in self.speakers2:
            sources2.extend(speaker.sources)

        sources3 = []
        for speaker in self.speakers3:
            sources3.extend(speaker.sources)

        self.features1 = []
        for source in sources1:
            item = (source[0].joinpath(source[1]), class_to_idx1[source[2]])
            self.features1.append(item)

        self.features2 = []
        for source in sources2:
            item = (source[0].joinpath(source[1]), class_to_idx2[source[2]])
            self.features2.append(item)

        self.features3 = []
        for source in sources3:
            item = (source[0].joinpath(source[1]), class_to_idx3[source[2]])
            self.features3.append(item)

        mean1 = np.load(self.data_dir1.joinpath('mean.npy'))
        std1 = np.load(self.data_dir1.joinpath('std.npy'))

        mean2 = np.load(self.data_dir2.joinpath('mean.npy'))
        std2 = np.load(self.data_dir2.joinpath('std.npy'))

        mean3 = np.load(self.data_dir3.joinpath('mean.npy'))
        std3 = np.load(self.data_dir3.joinpath('std.npy'))

        self.transform1 = T.Compose([
            Normalize(mean1, std1),
            TimeReverse(),
        ])
        self.transform2 = T.Compose([
            Normalize(mean2, std2),
            TimeReverse(),
        ])
        self.transform3 = T.Compose([
            Normalize(mean3, std3),
            TimeReverse(),
        ])


    def load_feature(self, feature_path, speaker_id):
        feature = np.load(feature_path)

        if self.is_test:
            test_sequence = generate_test_sequence(
                feature, self.partial_n_frames)
            # print(test_sequence.shape, self.partial_n_frames)
            return test_sequence, speaker_id
        else:
            if feature.shape[0] <= self.partial_n_frames:
                start = 0
                while feature.shape[0] < self.partial_n_frames:
                    feature = np.repeat(feature, 2, axis=0)
            else:
                start = np.random.randint(
                    0, feature.shape[0] - self.partial_n_frames)
            end = start + self.partial_n_frames
            return feature[start:end], speaker_id

    def __getitem__(self, index):
        feature_path1, speaker_id1 = self.features1[index]
        feature1, speaker_id1 = self.load_feature(feature_path1, speaker_id1)

        feature_path2, speaker_id2 = self.features2[index]
        feature2, speaker_id2 = self.load_feature(feature_path2, speaker_id2)

        feature_path3, speaker_id3 = self.features3[index]
        feature3, speaker_id3 = self.load_feature(feature_path3, speaker_id3)

        if self.transform1 is not None:
            feature1 = self.transform1(feature1)

        if self.transform2 is not None:
            feature2 = self.transform2(feature2)

        if self.transform3 is not None:
            feature3 = self.transform3(feature3)

        return feature1, speaker_id1, feature2, speaker_id2, feature3, speaker_id3,{feature_path1:[]}

    def __len__(self):
        return min(len(self.features1),len(self.features2),len(self.features3))
