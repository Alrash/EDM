# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm
from mix import mixup_distributions


class SAGM(Algorithm):
    def __init__(self, args):
        super(SAGM, self).__init__(args)
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

        def loss_fn(predictions, targets):
            return F.cross_entropy(predictions, targets)

        if self.args.level in [1]:
            aug, self.mu, self.std = mixup_distributions(all_x, all_d, mode='inter', momentum=self.args.mix_momentum,
                                                       mu_hist=self.mu, std_hist=self.std, alpha=self.args.mix_alpha)
            all_x, all_y = torch.cat([all_x, aug]), torch.cat([all_y, all_y])

        if self.args.level in [2]:
            aug, self.mu, self.std = mixup_distributions(all_x, all_d, mode='extra', alpha=self.args.mix_alpha,
                                                       momentum=self.args.mix_momentum if self.args.level == 2 or self.args.mix_momentum == 0 else 1,
                                                       mu_hist=self.mu, std_hist=self.std, ex_alpha=self.args.mix_ex_alpha)
            all_x, all_y = torch.cat([all_x, aug]), torch.cat([all_y, all_y])

        opt.set_closure(loss_fn, all_x, all_y)
        predictions, loss = opt.step()
        sch.step()
        opt.update_rho_t()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)

    def get_feature(self, x):
        return self.featurizer(x)