# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import Adver_network, common_network
from alg.algs.base import Algorithm


class CrossGrad(Algorithm):

    def __init__(self, args):

        super(CrossGrad, self).__init__(args)

        self.featurizer = get_fea(args)
        self.classifier = common_network.feat_classifier(
            args.num_classes, self.featurizer.in_features, args.classifier)
        self.discriminator = Adver_network.Discriminator(
            self.featurizer.in_features, args.dis_hidden, args.domain_num - len(args.test_envs))
        self.args = args

        self.EPS_F = 1.0  # scaling parameter for D's gradients
        self.EPS_D = 1.0  # scaling parameter for F's gradients
        self.ALPHA_F = 0.5  # balancing weight for the label net's loss
        self.ALPHA_D = 0.5

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        all_d = torch.cat([data[2].cuda().long() for data in minibatches])

        all_x.requires_grad = True

        loss_d = F.cross_entropy(self.discriminator(self.featurizer(all_x)), all_d)
        loss_d.backward()
        grad_d = torch.clamp(all_x.grad.data, min=-0.1, max=0.1)
        x_d = all_x.data + self.EPS_D * grad_d

        all_x.grad.data.zero_()
        loss_f = F.cross_entropy(self.classifier(self.featurizer(all_x)), all_y)
        loss_f.backward()
        grad_f = torch.clamp(all_x.grad.data, min=-0.1, max=0.1)
        x_f = all_x.data + self.EPS_F * grad_f

        all_x = all_x.detach()

        loss_f1 = F.cross_entropy(self.classifier(self.featurizer(all_x)), all_y)
        loss_f2 = F.cross_entropy(self.classifier(self.featurizer(x_f)), all_y)
        loss_f = (1 - self.ALPHA_F) * loss_f1 + self.ALPHA_F * loss_f2

        loss_d1 = F.cross_entropy(self.discriminator(self.featurizer(all_x)), all_d)
        loss_d2 = F.cross_entropy(self.discriminator(self.featurizer(x_d)), all_d)
        loss_d = (1 - self.ALPHA_D) * loss_d1 + self.ALPHA_D * loss_d2

        loss = loss_f + loss_d
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'total': loss.item(), 'class': loss_f.item(), 'dis': loss_d.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))
