# coding=utf-8
import numpy as np
import torch
import torch.nn.functional as F

from datautil.util import random_pairs_of_minibatches
from alg.algs.ERM import ERM
from mix import mixup_distributions


class Mixup(ERM):
    def __init__(self, args):
        super(Mixup, self).__init__(args)
        self.args = args

    def update(self, minibatches, opt, sch):
        objective = 0

        all_x = torch.cat([data[0] for data in minibatches])
        all_y = torch.cat([data[1] for data in minibatches])
        all_d = torch.cat([data[2] for data in minibatches])
        aug = None

        if self.args.level in [1]:
            aug, self.mu, self.std = mixup_distributions(all_x, all_d, mode='inter', momentum=self.args.mix_momentum,
                                                       mu_hist=self.mu, std_hist=self.std, alpha=self.args.mix_alpha)

        if self.args.level in [2]:
            aug, self.mu, self.std = mixup_distributions(all_x, all_d, mode='extra', alpha=self.args.mix_alpha,
                                                       momentum=self.args.mix_momentum if self.args.level == 2 or self.args.mix_momentum == 0 else 1,
                                                       mu_hist=self.mu, std_hist=self.std, ex_alpha=self.args.mix_ex_alpha)

        if aug is not None:
            index = torch.randperm(aug.size(0)).long()
            aug = [aug[index], all_y[index], all_d[index]]

        for (xi, yi, di), (xj, yj, dj) in random_pairs_of_minibatches(self.args, minibatches, aug):
            lam = np.random.beta(self.args.mixupalpha, self.args.mixupalpha)

            x = (lam * xi + (1 - lam) * xj).cuda().float()

            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi.cuda().long())
            objective += (1 - lam) * \
                F.cross_entropy(predictions, yj.cuda().long())

        objective /= len(minibatches)

        opt.zero_grad()
        objective.backward()
        opt.step()
        if sch:
            sch.step()
        return {'class': objective.item()}
