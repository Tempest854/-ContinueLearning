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


class DGCERCIL(ContinualModel):
    NAME = 'dgc_er_cil'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(DGCERCIL, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.DGC = DGC(self.parameters, self.loss, self.device)

    def observe(self, inputs, labels, not_aug_inputs):
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

        self.opt.step()

        return loss.item()

    def end_task(self, dataset):
        self.net.eval()
        self.net.zero_grad()
        for i, data in enumerate(dataset.train_loader):
            inputs, labels, not_aug_inputs = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            not_aug_inputs = not_aug_inputs.to(self.device)
            self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        if self.args.DGC:
            self.DGC.update_history(self.net, dataset.train_loader)