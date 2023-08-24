import torch
import numpy as np
from collections import Iterable


__all__ = ['projection', 'mixup_distributions', 'mixup_two_distributions']


# (x - mu) / std * mixed_std + mixed_mu
def projection(x, mu, std, mixed_mu, mixed_std):
    return torch.mul(torch.div(x - mu, std), mixed_std) + mixed_mu


# input:
#   x -> N x C x H x W or N x dim
#   d -> N
# output:
#   mu: dist -> {domain: mean -> C}
#   std: dist -> {domain: std -> C}
def calc_normal_distribution(x: torch.Tensor, d: torch.Tensor):
    mu, std, axis = {}, {}, [0, 2, 3] if len(x.size()) == 4 else [0]
    for domain in d.unique():
        mu[int(domain)] = x[d == domain, ...].mean(axis = axis, keepdim = True)
        std[int(domain)] = x[d == domain, ...].std(axis = axis, keepdim = True)
    return mu, std


def calc_momentum_distribution_parameters(x: torch.Tensor, d: torch.Tensor, momentum, mu_hist, std_hist):
    mu, std = calc_normal_distribution(x, d)
    if momentum > 0:
        for domain in mu.keys():
            mu[int(domain)] = momentum * mu_hist[int(domain)] + (1 - momentum) * mu[int(domain)] \
                if mu_hist is not None and mu_hist[int(domain)] is not None else mu[int(domain)]
            std[int(domain)] = momentum * std_hist[int(domain)] + (1 - momentum) * std[int(domain)] \
                if std_hist is not None and std_hist[int(domain)] is not None else std[int(domain)]
    return mu, std


# mixup mean and sigma
def mix_distributions(mu: dict, std: dict, lmd):
    domain = list(mu.keys())[0]
    mu_new = torch.zeros_like(mu[domain], device = mu[domain].device)
    std_new = torch.zeros_like(std[domain], device = std[domain].device)
    for index, domain in enumerate(mu.keys()):
        mu_new += lmd[index] * mu[domain]
        std_new += (lmd[index] ** 2) * torch.pow(std[domain], 2)
    std_new = torch.sqrt(std_new)
    return mu_new, std_new


def check_mixup_kwargs(**kwargs) -> bool:
    if kwargs is not None and ('alpha' in kwargs.keys() or 'lmd' in kwargs.keys()):
        assert 'alpha' not in kwargs.keys() or kwargs['alpha'] > 0
        assert 'lmd' not in kwargs.keys() or 0 < kwargs['lmd'] < 1 or kwargs['lmd'] == -1 or isinstance(kwargs['lmd'], Iterable)
        return True
    else:
        return False


def check_mixup_mode(mode, **kwargs) -> bool:
    return mode in ['inter', 'extra'] and (kwargs is not None and 'ex_alpha' in kwargs.keys() if mode == 'extra' else True)


def check_mixup_momentum(momentum, mu, std) -> bool:
    return (0 < momentum <= 1 and mu is not None and std is not None) or momentum == 0


# output:
#   mean and std of new domain
def extrapolate_domain_distributions(mu, std, alpha, control = True):
    # random sampling from dirichlet distribution
    while True:
        lmd = torch.distributions.dirichlet.Dirichlet(alpha * torch.ones(len(mu.keys()))).sample().cpu().detach().numpy()
        # normalization
        lmd /= lmd.sum()
        if np.max(lmd) > 0.9:
            break

    # mix each domain
    idx, max_lmd = np.argmax(lmd), np.max(lmd)
    mu_new, std_new = {}, {}
    for index, domain in enumerate(mu.keys()):
        l = lmd.copy()
        if control:
            l[[idx, index]] = l[[index, idx]]
            l /= -max_lmd
            l[index] = 1. / max_lmd
        else:
            l /= -l[index]
            l[index] = 1. / lmd[index]

        # obtain new distribution by mix
        mu_new[domain], std_new[domain] = mix_distributions(mu, std, l)
    # end for domain
    return mu_new, std_new


def mixup_distributions(x: torch.Tensor, d: torch.Tensor, mode: str = 'inter', momentum = 0, mu_hist = None, std_hist = None, **kwargs):
    assert check_mixup_mode(mode, **kwargs)
    assert check_mixup_momentum(momentum, mu_hist, std_hist)
    assert check_mixup_kwargs(**kwargs)

    # calculate mean and std
    mu, std = calc_momentum_distribution_parameters(x, d, momentum, mu_hist, std_hist)

    if mode == 'extra':
        mu_new, std_new = extrapolate_domain_distributions(mu, std, alpha = kwargs['ex_alpha'])
    else:
        mu_new, std_new = mu, std

    aug, D = torch.empty_like(x, device = x.device), d.unique().size(0)
    # create lmd and then normalize it
    if 'lmd' in kwargs.keys():
        lmd = 1. / D * torch.ones(D) if kwargs['lmd'] == -1 else torch.Tensor(kwargs['lmd'])
    else:
        lmd = torch.distributions.dirichlet.Dirichlet(kwargs['alpha'] * torch.ones(D)).sample()
    lmd = lmd.cpu().detach().numpy()
    lmd /= lmd.sum()

    # mixed distribution
    mixed_mu, mixed_std = mix_distributions(mu_new, std_new, lmd)

    # project to mixed distribution
    for domain in d.unique():
        aug[d == domain, ...] = projection(x[d == domain, ...], mu[int(domain)], std[int(domain)], mixed_mu, mixed_std)
    # end for projection

    if momentum == 0:
        mu, std = None, None

    return aug, mu, std


def mixup_two_distributions(x: torch.Tensor, d: torch.Tensor, mode: str = 'inter', momentum = 0, mu_hist = None, std_hist = None, **kwargs):
    assert check_mixup_mode(mode, **kwargs)
    assert check_mixup_momentum(momentum, mu_hist, std_hist)
    assert check_mixup_kwargs(**kwargs)

    mu, std = calc_momentum_distribution_parameters(x, d, momentum, mu_hist, std_hist)

    if mode == 'extra':
        ex_lmd = torch.distributions.Beta(kwargs['ex_alpha'], kwargs['ex_alpha']).sample().cpu().detach().numpy()
        if 'control' in kwargs.keys() and kwargs['control']:
            ex_lmd = 1 - ex_lmd if ex_lmd < 0.5 else ex_lmd
        ex_lmd = [1. / ex_lmd, 1 - 1. / ex_lmd]

    if 'lmd' in kwargs.keys():
        lmd = 0.5 if kwargs['lmd'] == -1 else kwargs['lmd']
    else:
        lmd = torch.distributions.Beta(kwargs['alpha'], kwargs['alpha']).sample().cpu().detach().numpy()

    domain_list, idx, aug = d.unique(), np.arange(d.size(0)), torch.zeros_like(x, device = x.device)
    for domain in domain_list:
        # random choice other domain
        num, idx_d = d[d == domain].size(0), idx[d == domain]
        leave_domain = domain_list[domain_list != domain]
        random_domain = leave_domain[np.random.randint(0, leave_domain.size(0), num)]
        for mixed_domain in random_domain.unique():
            # extrapolate distribution
            if mode == 'extra':
                ex_l = np.zeros(len(mu.keys()))
                ex_l[[int(domain), int(mixed_domain)]] = ex_lmd
                mu_new, std_new = mix_distributions(mu, std, ex_l)
            else:
                mu_new, std_new = mu, std

            l = np.zeros(len(mu.keys()))
            l[[int(domain), int(mixed_domain)]] = [lmd, 1 - lmd]
            mixed_mu, mixed_std = mix_distributions(mu_new, std_new, l)
            aug[idx_d[random_domain == mixed_domain], ...] = projection(
                x[idx_d[random_domain == mixed_domain]], mu[int(domain)], std[int(domain)], mixed_mu, mixed_std)
        # end for domain_j
    # end for domain_i

    if momentum == 0:
        mu, std = None, None

    return aug, mu, std


if __name__ == '__main__':
    import time
    n = 500
    for i in range(100):
        start = time.time()
        img, mu, var = mixup_distributions(torch.randn(n, 3, 224, 224), d = torch.from_numpy(np.random.randint(0, 3, n)), alpha = 0.5, mode = 'extra', ex_alpha=0.5)
        print(time.time() - start)
    # mu = {0: torch.tensor([1., 0]), 1: torch.tensor([-1., 0]), 2: torch.tensor([0, np.sqrt(3)])}
    # std = {0: torch.tensor([1.]), 1: torch.tensor([2.]), 2: torch.tensor([np.sqrt(3)])}
    # extrapolate_domain_distributions(mu, std, 0.5)
    print("ok")
