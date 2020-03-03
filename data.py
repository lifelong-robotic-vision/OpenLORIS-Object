import copy
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset
import torch
import glob
from PIL import Image
import os

my_transform = transforms.Compose([transforms.ToTensor()])
datapath = ['bottle_01', 'bottle_02', 'bottle_03', 'bottle_04', 'bowl_01', 'bowl_02', 'bowl_03', 'bowl_04', 'bowl_05',
            'corkscrew_01', 'cottonswab_01', 'cottonswab_02', 'cup_01', 'cup_02', 'cup_03', 'cup_04', 'cup_05',
            'cup_06', 'cup_07', 'cup_08', 'cup_10', 'cushion_01', 'cushion_02', 'cushion_03', 'glasses_01',
            'glasses_02', 'glasses_03', 'glasses_04', 'knife_01', 'ladle_01', 'ladle_02', 'ladle_03', 'ladle_04',
            'mask_01', 'mask_02', 'mask_03', 'mask_04', 'mask_05', 'paper_cutter_01', 'paper_cutter_02',
            'paper_cutter_03', 'paper_cutter_04', 'pencil_01', 'pencil_02', 'pencil_03', 'pencil_04', 'pencil_05',
            'plasticbag_01', 'plasticbag_02', 'plasticbag_03', 'plug_01', 'plug_02', 'plug_03', 'plug_04', 'pot_01',
            'scissors_01', 'scissors_02', 'scissors_03', 'stapler_01', 'stapler_02', 'stapler_03', 'thermometer_01',
            'thermometer_02', 'thermometer_03', 'toy_01', 'toy_02', 'toy_03', 'toy_04', 'toy_05','nail_clippers_01','nail_clippers_02',
            'nail_clippers_03', 'bracelet_01', 'bracelet_02','bracelet_03', 'comb_01','comb_02',
            'comb_03', 'umbrella_01','umbrella_02','umbrella_03','socks_01','socks_02','socks_03',
            'toothpaste_01','toothpaste_02','toothpaste_03','wallet_01','wallet_02','wallet_03',
            'headphone_01','headphone_02','headphone_03', 'key_01','key_02','key_03',
             'battery_01', 'battery_02', 'mouse_01', 'pencilcase_01', 'pencilcase_02', 'tape_01',
             'chopsticks_01', 'chopsticks_02', 'chopsticks_03',
               'notebook_01', 'notebook_02', 'notebook_03',
               'spoon_01', 'spoon_02', 'spoon_03',
               'tissue_01', 'tissue_02', 'tissue_03',
              'clamp_01', 'clamp_02', 'hat_01', 'hat_02', 'u_disk_01', 'u_disk_02', 'swimming_glasses_01'
            ]


class MyDataset(Dataset):

    def __init__(self, batch_num, mode='train', own_transform=None, factor='clutter'):
        batch_num += 1
        self.transform = own_transform
        if mode == 'train':
            self.imgs = []
            self.labels = []
            for i in range(len(datapath)):
                temp = glob.glob('img/' + factor + '/train/task{}/{}/*.jpg'.format(batch_num, datapath[i]))

                self.imgs.extend([Image.open(x).convert('RGB').resize((50, 50)) for x in temp])
                self.labels.extend([i] * len(temp))
            print("  --> batch{}'-dataset consisting of {} samples".format(batch_num, len(self)))
        else:
            self.imgs = []
            self.labels = []
            for i in range(len(datapath)):
                temp = glob.glob('img/' + factor + '/test/task{}/{}/*.jpg'.format(batch_num, datapath[i]))
                self.imgs.extend([Image.open(x).convert('RGB').resize((50, 50)) for x in temp])
                self.labels.extend([i] * len(temp))
            print("  --> test'-dataset consisting of {} samples".format(len(self)))

    def __setitem__(self, index, value):
        self.imgs[index] = value[0]
        self.labels[index] = value[1]

    def __getitem__(self, index):

        fn = self.imgs[index]
        label = self.labels[index]
        img = fn

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):

        return len(self.imgs)


# ----------------------------------------------------------------------------------------------------------#


class SubDataset(Dataset):

    def __init__(self, original_dataset, sub_labels, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "train_labels"):
                if self.dataset.target_transform is None:
                    label = self.dataset.train_labels[index]
                else:
                    label = self.dataset.target_transform(self.dataset.train_labels[index])
            elif hasattr(self.dataset, "test_labels"):
                if self.dataset.target_transform is None:
                    label = self.dataset.test_labels[index]
                else:
                    label = self.dataset.target_transform(self.dataset.test_labels[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indeces.append(index)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample


class ExemplarDataset(Dataset):

    def __init__(self, exemplar_sets, target_transform=None):
        super().__init__()
        self.exemplar_sets = exemplar_sets
        self.target_transform = target_transform

    def __len__(self):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            total += len(self.exemplar_sets[class_id])
        return total

    def __getitem__(self, index):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            exemplars_in_this_class = len(self.exemplar_sets[class_id])
            if index < (total + exemplars_in_this_class):
                class_id_to_return = class_id if self.target_transform is None else self.target_transform(class_id)
                exemplar_id = index - total
                break
            else:
                total += exemplars_in_this_class
        image = torch.from_numpy(self.exemplar_sets[class_id][exemplar_id])
        return (image, class_id_to_return)


def get_multitask_experiment(name, scenario, tasks, only_config=False, verbose=False,
                             exception=False, factor='clutter'):
    if name == 'mydataset':
        classes_per_task = 121
        train_datasets = []
        test_datasets = []

        for i in range(tasks):
            train_datasets.append(MyDataset(i, mode='train', own_transform=my_transform, factor=factor))
            test_datasets.append(MyDataset(i, mode='test', own_transform=my_transform, factor=factor))
        config = {'size': 50, 'channels': 3, 'classes': 121}


    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    # If needed, update number of (total) classes in the config-dictionary
    config['classes'] = classes_per_task

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    return config if only_config else ((train_datasets, test_datasets), config, classes_per_task)
