import torch
import numpy as np
from copy import deepcopy


class DGC:
    def __init__(self, params, loss, device, update_every=200, gamma=1e-3):
        self.count = 0
        self.update_every = update_every
        self.gamma = gamma
        self.loss = loss
        self.device = device
        self.task_cnt = 0
        self.lastnet = None
        self.grad_dims = []
        for param in params():
            self.grad_dims.append(param.data.numel())
        self.history_grad = torch.zeros(np.sum(self.grad_dims)).to(self.device)
        self.record = torch.zeros(np.sum(self.grad_dims)).to(self.device)

    def update_and_replay(self, net, inputs, labels):
        self.count += 1
        self._cac_grad(net, inputs, labels)

        if self.count == self.update_every:
            self.history_grad += self.record / self.count / self.task_cnt
            self.count = 0
            self.record.fill_(0.0)
            self.lastnet = deepcopy(net)

        self.overwrite_grad(net.parameters, self.history_grad * self.gamma)

    def update_history(self, net, loader):
        if self.lastnet is None:
            self.lastnet = deepcopy(net)
        self.task_cnt += 1
        status = self.lastnet.training
        self.lastnet.eval()
        self.lastnet.zero_grad()
        for i, data in enumerate(loader):
            inputs, labels, not_aug_inputs = data
            not_aug_inputs, labels = not_aug_inputs.to(self.device), labels.to(self.device)
            outputs = self.lastnet(not_aug_inputs)
            loss = self.loss(outputs, labels)
            loss.backward()

        self._update_history(self.lastnet.parameters, len(loader))
        self.lastnet.zero_grad()
        self.lastnet.train(status)

    def update_history_batch(self, net, inputs, labels):
        if self.lastnet is None:
            self.lastnet = deepcopy(net)
        self.task_cnt += 1
        status = self.lastnet.training
        self.lastnet.eval()
        self.lastnet.zero_grad()

        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = self.lastnet(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()

        self._update_history(self.lastnet.parameters, 1)
        self.lastnet.zero_grad()
        self.lastnet.train(status)

    def _cac_grad(self, net, inputs, labels):
        status = net.training
        net.eval()
        net.zero_grad()
        outputs = net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        net.train(status)

        status = self.lastnet.training
        self.lastnet.eval()
        self.lastnet.eval()
        self.lastnet.zero_grad()
        outputs = self.lastnet(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.lastnet.train(status)

        count = 0
        for p1, p2 in zip(net.parameters(), self.lastnet.parameters()):
            if p1.grad is not None and p2.grad is not None:
                begin = 0 if count == 0 else sum(self.grad_dims[:count])
                end = np.sum(self.grad_dims[:count + 1])
                self.record[begin: end] += (p1.grad.data.view(-1) - p2.grad.data.view(-1))
            count += 1

        net.zero_grad()

    def _update_history(self, params, amt=1):
        self.history_grad *= (self.task_cnt - 1) / self.task_cnt
        count = 0
        for param in params():
            if param.grad is not None:
                begin = 0 if count == 0 else sum(self.grad_dims[:count])
                end = np.sum(self.grad_dims[:count + 1])
                self.history_grad[begin: end] += param.grad.data.view(-1) / amt / self.task_cnt
            count += 1

    def overwrite_grad(self, params, newgrad):
        count = 0
        for param in params():
            if param.grad is not None:
                begin = 0 if count == 0 else sum(self.grad_dims[:count])
                end = sum(self.grad_dims[:count + 1])
                this_grad = newgrad[begin: end].contiguous().view(
                    param.grad.data.size())
                param.grad.data.copy_(this_grad)
            count += 1