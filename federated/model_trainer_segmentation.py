import logging
from utils.weight_perturbation import WPOptim
import torch
import cv2
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from .model_trainer import ModelTrainer
from utils.loss import DiceLoss, entropy_loss,DiceLoss_test
from data.prostate.transforms import transforms_for_noise, transforms_for_rot, transforms_for_scale, transforms_back_scale, transforms_back_rot
import copy
import numpy as np
import random
import torch.optim as optim
from PIL import Image


def smooth_loss(output, d=10):
    output_pred = torch.nn.functional.softmax(output, dim=1)
    output_pred_foreground = output_pred[:,1:,:,:]
    m = nn.MaxPool2d(kernel_size=2*d+1, stride=1, padding=d)
    loss = (m(output_pred_foreground) + m(-output_pred_foreground))*(1e-3*1e-3)
    return loss


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss
    
def deterministic(seed):
     cudnn.benchmark = False
     cudnn.deterministic = True
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     random.seed(seed)

class ModelTrainerSegmentation(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = DiceLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, amsgrad=True)

        epoch_loss = []
        epoch_acc = []

        for epoch in range(args.wk_iters):
            batch_loss = []
            batch_acc = []
            for batch_idx, (x, labels) in enumerate(train_data):
                model.zero_grad()

                x, labels = x.to(device), labels.to(device)
                log_probs = model(x,'main')
                loss = criterion(log_probs, labels)
                acc = DiceLoss().dice_coef(log_probs, labels).item()

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                batch_acc.append(acc)

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_acc.append(sum(batch_acc) / len(batch_acc))
            logging.info('Client Index = {}\tEpoch: {}\tAcc:{:.4f}\tLoss: {:.4f}'.format(
            self.id, epoch, sum(epoch_acc) / len(epoch_acc),sum(epoch_loss) / len(epoch_loss)))

    def test(self, test_data, device, args):
        model = copy.deepcopy(self.model)

        model.to(device)
        model.eval()

        metrics = {
            'test_acc': 0,
            'test_loss': 0,
        }
        criterion = DiceLoss().to(device)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x, "main")
                loss = criterion(pred, target)
                acc_l = DiceLoss().dice_coef(pred, target)
                acc = acc_l.item()

                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_acc'] += acc

        metrics["test_loss"] = metrics["test_loss"] / len(test_data)
        metrics["test_acc"] = metrics["test_acc"] / len(test_data)

        return metrics
    
