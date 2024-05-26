import torch

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from models.utils.DGC import DGC


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--DGC', type=bool, default=False, help='Use DGC or not.')
    return parser


class DGCERTF(ContinualModel):
    NAME = 'dgc_er_tfcl'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(DGCERTF, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.DGC = DGC(self.parameters, self.loss, self.device)

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]
        self.opt.zero_grad()

        if self.args.DGC:
            if not self.buffer.is_empty():
                buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size)
                self.DGC.update_and_replay(self.net, buf_inputs, buf_labels)

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])
        
        self.opt.step()

        if self.args.DGC:
            self.DGC.update_history_batch(self.net, not_aug_inputs, labels[:real_batch_size])

        return loss.item()