# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import Adver_network, common_network
from alg.algs.base import Algorithm
from mix import mixup_distributions


class EntropyLoss(torch.nn.Module):
    """Entropy loss
    Arguments:
        t : temperature
    """

    def __init__(self, t=1.):
        super(EntropyLoss, self).__init__()
        self.t = t

    def forward(self, lbl, pred):
        """Compute loss.
        Arguments:
            lbl (torch.tensor:float): predictions, not confidence, not label.
            pred (torch.tensor:float): predictions.
        Returns:
            loss (torch.tensor:float): entropy loss
        """
        loss = torch.mean(torch.sum(F.softmax(lbl/self.t, dim=-1) * F.log_softmax(pred/self.t, dim=-1), dim=-1))
        return -loss


class DANN(Algorithm):

    def __init__(self, args):

        super(DANN, self).__init__(args)

        self.featurizer = get_fea(args)
        self.classifier = common_network.feat_classifier(
            args.num_classes, self.featurizer.in_features, args.classifier)
        self.discriminator = Adver_network.Discriminator(
            self.featurizer.in_features, args.dis_hidden, args.domain_num - len(args.test_envs))
        self.args = args

        self.num_domains = args.domain_num - len(args.test_envs)
        self.mu = {domain: None for domain in range(self.num_domains)}
        self.std = {domain: None for domain in range(self.num_domains)}

    def _calc_loss(self, z, y, d):
        disc_input = z
        disc_input = Adver_network.ReverseLayerF.apply(
            disc_input, self.args.alpha)
        disc_out = self.discriminator(disc_input)
        disc_loss = F.cross_entropy(disc_out, d) if self.args.adv_hard else EntropyLoss()(disc_out, disc_out)

        pred = self.classifier(z)
        cls_loss = F.cross_entropy(pred, y)
        return cls_loss + disc_loss, cls_loss, disc_loss

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        disc_labels = torch.cat([
            torch.full((data[0].shape[0], ), i,
                       dtype=torch.int64, device='cuda')
            for i, data in enumerate(minibatches)
        ])

        all_z = self.featurizer(all_x)
        loss, classifier_loss, disc_loss = self._calc_loss(all_z, all_y, disc_labels)

        if self.args.level in [1]:
            x, self.mu, self.std = mixup_distributions(all_x, disc_labels, mode='inter', momentum=self.args.mix_momentum,
                                                       mu_hist=self.mu, std_hist=self.std, alpha=self.args.mix_alpha)
            z = self.featurizer(x)
            mix_loss, mix_classifier_loss, mix_disc_loss = self._calc_loss(z, all_y, disc_labels)
            loss, classifier_loss, disc_loss = loss + mix_loss, classifier_loss + mix_classifier_loss, disc_loss + mix_disc_loss

        if self.args.level in [2]:
            x, self.mu, self.std = mixup_distributions(all_x, disc_labels, mode='extra', alpha=self.args.mix_alpha,
                                                       momentum=self.args.mix_momentum if self.args.level == 2 or self.args.mix_momentum == 0 else 1,
                                                       mu_hist=self.mu, std_hist=self.std, ex_alpha=self.args.mix_ex_alpha)
            z = self.featurizer(x)
            mix_loss, mix_classifier_loss, mix_disc_loss = self._calc_loss(z, all_y, disc_labels)
            loss, classifier_loss, disc_loss = loss + mix_loss, classifier_loss + mix_classifier_loss, disc_loss + mix_disc_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'total': loss.item(), 'class': classifier_loss.item(), 'dis': disc_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

    def get_feature(self, x):
        return self.featurizer(x)
