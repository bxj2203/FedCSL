import copy
import logging
import random
import sys, os
import torch
import numpy as np
from .client import Client
import re
import os
import sys
from tqdm import tqdm
import nibabel as nib
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch.nn.functional as F
import SimpleITK as sitk

class FedAvgAPI(object):
    def __init__(self, dataset, device, args, model_trainer):
        """
        dataset: data loaders and data size info
        """
        self.device = device
        self.args = args
        client_num, [train_data_num, val_data_num, test_data_num, train_data_local_num_dict, train_data_local_dict,
                     val_data_local_dict, test_data_local_dict, ood_data] = dataset
        self.client_num_in_total = client_num
        self.client_num_per_round = int(self.client_num_in_total * self.args.percent)
        self.train_data_num_in_total = train_data_num
        self.val_data_num_in_total = val_data_num
        self.test_data_num_in_total = test_data_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.val_data_local_dict = val_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.ood_data = ood_data

        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, val_data_local_dict, test_data_local_dict,
                            model_trainer)
        logging.info("############setup ood clients#############")
        self.ood_client = Client(-1, None, None, ood_data, len(ood_data.dataset), self.args, self.device, model_trainer)
        self.best_profmance = 0
        self.ood_performance = {"before": []}



    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, val_data_local_dict,
                       test_data_local_dict, model_trainer):
        logging.info("############setup inner clients#############")
        for client_idx in range(self.client_num_in_total):
            c = Client(client_idx, train_data_local_dict[client_idx], val_data_local_dict[client_idx],
                       test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)

    def calculate_lesion_volume(self,lesion_mask_path):

        lesion_mask = sitk.ReadImage(lesion_mask_path)
        spacing = lesion_mask.GetSpacing()

        lesion_mask_array = sitk.GetArrayFromImage(lesion_mask)
        binary_lesion_mask_array = np.zeros_like(lesion_mask_array)
        binary_lesion_mask_array[lesion_mask_array > 0] = 1

        voxel_volume = spacing[0] * spacing[1] * spacing[2]

        lesion_volume = np.sum(binary_lesion_mask_array) * voxel_volume
        print(np.sum(binary_lesion_mask_array))
        return lesion_volume

    def compute_snr(self,image_path, mask_path):

        image_nii = nib.load(image_path)
        mask_nii = nib.load(mask_path)

        image_data = image_nii.get_fdata()
        mask_data = mask_nii.get_fdata()
        MIN_BOUND = -1200
        MAX_BOUND = 200
        image_data = (image_data - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        roi = image_data[mask_data > 0]

        signal_mean = np.mean(roi)
        noise_std = np.std(roi)

        snr = signal_mean / noise_std

        return snr

    def train(self):
        w_global = self.model_trainer.get_model_params()
        for round_idx in range(self.args.comm_round):

            logging.info("============ Communication round : {}".format(round_idx))

            w_locals,SNR,lesion_volume = [],[],[]

            client_indexes = self._client_sampling(round_idx, self.client_num_in_total,
                                                   self.client_num_per_round)
            logging.info("client_indexes = " + str(client_indexes))
            for idx, client in enumerate(self.client_list):

                client_idx = client_indexes[idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.val_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])

                w = client.train(copy.deepcopy(w_global))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                SNR.append(self.compute_snr('client_path'))
                lesion_volume.append(self.calculate_lesion_volume('client_path'))
            quality = [SNR[i]*lesion_volume[i]/sum(SNR*lesion_volume) for i in range(len(SNR))]
            w_global = self._aggregate(w_locals,round_idx,self.args.comm_round,quality)
            self.model_trainer.set_model_params(w_global)

            val_metrics = self._local_val_on_all_clients(round_idx)

            if sum(val_metrics) / len(val_metrics) > self.best_profmance:
                test_metrics = self._local_test_on_all_clients(round_idx)
                print('*******************'*2,sum(test_metrics)/len(test_metrics))
                self.best_profmance = sum(val_metrics) / len(val_metrics)
                torch.save(w_global, os.path.join(self.args.save_path,
                                                  "{}_global_round{}_{}".format(self.args.mode, round_idx,
                                                                                sum(test_metrics)/len(test_metrics))))


    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes



    def _aggregate(self, w_locals, round_idx,comm_round,quality):

        normalized_scores = torch.tensor(quality)

        temperature = (comm_round - round_idx) / comm_round
        soft_weights = [F.sigmoid((score - np.mean(quality)) * temperature / np.std(quality)) for score in
                        normalized_scores]

        soft_weights_normalized = [weight / sum(soft_weights) for weight in soft_weights]
        all_weight = []
        for weight in soft_weights_normalized:
            all_weight.append(weight.item())
        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                if i == 0:
                    w = all_weight[0]
                elif i == 1:
                    w = all_weight[1]
                else:
                    w = all_weight[2]
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params


    def _local_val_on_all_clients(self, round_idx):
        logging.info("============ local_validation_on_all_clients : {}".format(round_idx))

        val_metrics = {
            'acc': [],
            'losses': []
        }

        for client_idx in range(self.client_num_in_total):
            if self.val_data_local_dict[client_idx] is None:
                continue
            client = self.client_list[client_idx]
            client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                        self.val_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            local_metrics = client.local_validate()
            local_metrics_by_trajectory = client.local_validate_by_trajectory()


            val_metrics['acc'].append(copy.deepcopy(local_metrics['test_acc']))
            val_metrics['losses'].append(copy.deepcopy(local_metrics_by_trajectory['test_loss']))
            logging.info('Client Index = {}\tAcc:{:.4f}\tLoss: {:.4f}'.format(
                client_idx, local_metrics['test_acc'], local_metrics_by_trajectory['test_loss']))
        return  val_metrics['acc']

    def _local_test_on_all_clients(self, round_idx):
        logging.info("============ local_test_on_all_clients : {}".format(round_idx))

        test_metrics = {
            'acc': [],
            'losses': []
        }

        for client_idx in range(self.client_num_in_total):
            if self.test_data_local_dict[client_idx] is None:
                continue
            client = self.client_list[client_idx]
            client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                        self.val_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            test_local_metrics = client.local_test(True)

            test_metrics['acc'].append(copy.deepcopy(test_local_metrics['test_acc']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))
            logging.info('Client Index = {}\tAcc:{:.4f}\tLoss: {:.4f}'.format(
                client_idx, test_local_metrics['test_acc'], test_local_metrics['test_loss']))
        print('---------------------{}---------------------'.format(sum(test_metrics['acc'])/len(test_metrics['acc'])))
        return test_metrics['acc']

    def all_clients(self, round_idx,flag):
        logging.info("============ local_test_on_all_clients : {}".format(round_idx))

        test_metrics = {
            'acc': [],
            'losses': []
        }



        client = self.client_list[flag]
        client.update_local_dataset(flag, self.train_data_local_dict[flag],
                                    self.val_data_local_dict[flag],
                                    self.test_data_local_dict[flag],
                                    self.train_data_local_num_dict[flag])
        test_local_metrics = client.local_test(True)


        test_metrics['acc'].append(copy.deepcopy(test_local_metrics['test_acc']))
        test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))
        logging.info('Client Index = {}\tAcc:{:.4f}\tLoss: {:.4f}'.format(
            flag, test_local_metrics['test_acc'], test_local_metrics['test_loss']))
        print('---------------------{}---------------------'.format(sum(test_metrics['acc'])/len(test_metrics['acc'])))
        return test_metrics['acc']



