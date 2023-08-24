# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm
from mix import mixup_distributions


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, args):
        super(ERM, self).__init__(args)
        self.featurizer = get_fea(args)
        self.classifier = common_network.feat_classifier(
            args.num_classes, self.featurizer.in_features, args.classifier)

        self.network = nn.Sequential(
            self.featurizer, self.classifier)

        self.args = args
        self.num_domains = args.domain_num - len(args.test_envs)
        self.mu = {domain: None for domain in range(self.num_domains)}
        self.std = {domain: None for domain in range(self.num_domains)}

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        all_d = torch.cat([data[2].cuda().long() for data in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        if self.args.level in [1]:
            x, self.mu, self.std = mixup_distributions(all_x, all_d, mode='inter', momentum=self.args.mix_momentum,
                                                       mu_hist=self.mu, std_hist=self.std, alpha=self.args.mix_alpha)
            loss += F.cross_entropy(self.predict(x), all_y)

        if self.args.level in [2]:
            x, self.mu, self.std = mixup_distributions(all_x, all_d, mode='extra', alpha=self.args.mix_alpha,
                                                       momentum=self.args.mix_momentum if self.args.level == 2 or self.args.mix_momentum == 0 else 1,
                                                       mu_hist=self.mu, std_hist=self.std, ex_alpha=self.args.mix_ex_alpha)
            loss += F.cross_entropy(self.predict(x), all_y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'class': loss.item()}

    def predict(self, x):
        return self.network(x)

    def get_feature(self, x):
        return self.featurizer(x)