import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import configs.config as config

import lib.utils as utils



class AverageMeterWriter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        index_to_class_dict = utils.load_json_file(
            os.path.join(r"./lib/dataloaders/index_to_class", config.dataset_name + ".json"))
        self.label_names = list(index_to_class_dict.values())
        self.data = self.create_data_structure()
        self.reset()


    def create_data_structure(self, ):
        data = dict((label, 0.0) for label in self.label_names)
        data['count'] = 0
        data['dsc'] = 0.0
        return data


    def reset(self):
        for i in range(len(self.label_names)):
            self.data[self.label_names[i]] = 0.0
        self.data['dsc'] = 0.0
        self.data['count'] = 0


    def update(self, per_channel_dice):
        dice_coeff = np.mean(per_channel_dice)

        self.data['dsc'] += dice_coeff
        self.data['count'] += 1

        for i, score in enumerate(per_channel_dice):
            self.data[self.label_names[i]] += score

    def display_terminal(self):
        print_info = "DSC:{:.4f}".format(self.data["dsc"]/self.data['count'])
        for i, label_name in enumerate(self.label_names):
            print_info += "\t{}:{:.4f}".format(label_name,
                                               self.data[label_name] / self.data['count'])
        print(print_info)














