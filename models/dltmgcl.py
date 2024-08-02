import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from copy import deepcopy
from torch import nn
from torch.nn import functional as F
from backbone.MNISTMLP import L2LMLP


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Complementary Learning Systems Based Experience Replay')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    # Consistency Regularization Weight
    parser.add_argument('--reg_weight', type=float, default=0.1)

    # Stable Model parameters
    parser.add_argument('--neocortex_model_update_freq', type=float, default=0.70)
    parser.add_argument('--neocortex_model_alpha', type=float, default=0.999)

    # Plastic Model Parameters
    parser.add_argument('--hippocampus_model_update_freq', type=float, default=0.90)
    parser.add_argument('--hippocampus_model_alpha', type=float, default=0.999)

    return parser


# =============================================================================
# Mean-ER
# =============================================================================
class DLTMGCL(ContinualModel):
    NAME = 'dltmgcl'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(DLTMGCL, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        # Initialize hippocampus and neocortex model
        self.net_num = 50
        self.hippocampus_model = deepcopy(self.net).to(self.device)
        self.neocortex_model = [deepcopy(self.net).to(self.device)]

        # set regularization weight
        self.reg_weight = args.reg_weight
        # set parameters for hippocampus model
        self.hippocampus_model_update_freq = args.hippocampus_model_update_freq
        self.hippocampus_model_alpha = args.hippocampus_model_alpha
        # set parameters for neocortex model
        self.neocortex_model_update_freq = args.neocortex_model_update_freq
        self.neocortex_model_alpha = args.neocortex_model_alpha

        self.statics = {}

        self.consistency_loss = nn.MSELoss(reduction='none')
        self.current_task = 0
        self.global_step = 0
        self.neocortex_model_number = 1

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        loss = 0


        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            buf_inputs, buf_labels = buf_inputs.to(self.device), buf_labels.to(self.device)

            max_prob = torch.zeros(buf_labels.shape[0], device=self.device)
            max_logits = torch.zeros_like(self.neocortex_model[0](buf_inputs))

            for i in range(len(self.neocortex_model)):
                neocortex_model_logits = self.neocortex_model[i](buf_inputs)
                neocortex_model_prob = F.softmax(neocortex_model_logits, 1)
                label_mask = F.one_hot(buf_labels, num_classes=neocortex_model_logits.shape[-1]) > 0

                neocortex_model_prob_masked = neocortex_model_prob[label_mask]
                higher_prob_mask = neocortex_model_prob_masked > max_prob
                max_prob[higher_prob_mask] = neocortex_model_prob_masked[higher_prob_mask]
                max_logits[higher_prob_mask] = neocortex_model_logits[higher_prob_mask]

            l_cons = torch.mean(self.consistency_loss(self.net(buf_inputs), max_logits.detach()))
            l_reg = self.args.reg_weight * l_cons
            loss += l_reg

            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        ce_loss = self.loss(outputs, labels)
        loss += ce_loss

        # Log values
        if hasattr(self, 'writer'):
            self.writer.add_scalar(f'Task {self.current_task}/ce_loss', ce_loss.item(), self.iteration)
            self.writer.add_scalar(f'Task {self.current_task}/loss', loss.item(), self.iteration)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(not_aug_inputs, labels[:real_batch_size])

        # Update the neocortex model
        self.global_step += 1
        if torch.rand(1) < self.hippocampus_model_update_freq:
            self.update_hippocampus_model_variables()
        if torch.rand(1) < self.neocortex_model_update_freq:

            self.update_one_neocortex_model_variables()


        return loss.item()

    def update_hippocampus_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.hippocampus_model_alpha)
        for ema_param, param in zip(self.hippocampus_model.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def update_one_neocortex_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1),  self.neocortex_model_alpha)
        for ema_param, param in zip(self.neocortex_model[0].parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def update_neocortex_model_variables(self):
        if self.global_step // self.net_num - self.neocortex_model_number >= 0:
            for i in range(1, self.global_step // self.net_num - self.neocortex_model_number + 1):
                self.neocortex_model.append(deepcopy(self.neocortex_model[i - 1]).to(self.device))
            self.neocortex_model_number += self.global_step // self.net_num - self.neocortex_model_number
        alpha = min(1 - 1 / (self.global_step + 1), self.neocortex_model_alpha)
        for ema_param, param in zip(self.neocortex_model[self.global_step // self.net_num - 1].parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


    def copy_neocortex_model_variables(self):
        if self.global_step // self.net_num - self.neocortex_model_number >= 0:
            for i in range(1, self.global_step // self.net_num - self.neocortex_model_number + 1):
                self.neocortex_model.append(deepcopy(self.neocortex_model[i - 1]).to(self.device))
            self.neocortex_model_number += self.global_step // self.net_num - self.neocortex_model_number
        for ema_param, param in zip(self.neocortex_model[self.global_step // self.net_num - 1].parameters(), self.net.parameters()):
            ema_param = param.clone()

