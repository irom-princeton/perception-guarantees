import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def KLDiv_gaussian(mu1, var1, mu2, var2, var_is_logvar=True):
    if var_is_logvar:
        var1 = torch.exp(var1)
        var2 = torch.exp(var2)

    mu1 = torch.flatten(mu1)  # make sure we are 1xd so torch functions work as expected
    var1 = torch.flatten(var1)
    mu2 = torch.flatten(mu2)
    var2 = torch.flatten(var2)

    kl_div = 1/2 * torch.log(torch.div(var2, var1))
    kl_div += 1/2 * torch.div(var1 + torch.pow(mu2 - mu1, 2), var2)
    kl_div -= 1/2  # one for each dimension

    return torch.sum(kl_div)

class StochasticLayer(nn.Module):
    def __init__(self, weights_size, mu0=0):
        super().__init__()
        self.weights_size = weights_size

        self.mu = nn.Parameter(torch.zeros(weights_size))
        self.logvar = nn.Parameter(torch.zeros(weights_size))

        self.init_mu(mu0)
        self.init_logvar()

        self.stdev_xi = None

    def init_mu(self, mu=None, b_mu=None):
        n = self.mu.numel()
        stdev = math.sqrt(1./n)
        if mu is None:
            self.mu.data.uniform_(-stdev, stdev)
        else:
            self.mu.data += mu

    def init_logvar(self, logvar=0., b_logvar=0.):
        self.logvar.data.zero_()
        self.logvar.data += logvar

    def init_xi(self):
        stdev = torch.exp(0.5 * self.logvar)
        xi = stdev.data.new(stdev.size()).normal_(0, 1)
        self.stdev_xi = stdev * xi

    def forward(self, x):
        # x: shape of the output
        rand_sample = torch.randn(x).to(device=self.mu.device)
        output = self.mu + torch.exp(0.5 * self.logvar) * rand_sample
        return output

    def to_str(self):
        print("mu", self.mu.data.flatten()[:5].to('cpu').numpy())

    def calc_kl_div(self, prior):
        mu1 = self.mu
        logvar1 = self.logvar
        mu2 = prior.mu.clone().detach()
        logvar2 = prior.logvar.clone().detach()
        kl_div = KLDiv_gaussian(mu1, logvar1, mu2, logvar2, var_is_logvar=True)

        return kl_div


class StochasticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.compatible_classes = (StochasticLayer,
                                   )

    def forward(self, x):
        raise NotImplementedError()

    def init_xi(self, *args, **kwargs):
        for name, layer in self.named_modules():
            if layer.__class__ in self.compatible_classes:
                layer.init_xi(*args, **kwargs)

    def to_str(self, *args, **kwargs):
        for name, layer in self.named_modules():
            if layer.__class__ in self.compatible_classes:
                layer.to_str(*args, **kwargs)

    def init_logvar(self, *args, **kwargs):
        for name, layer in self.named_modules():
            if layer.__class__ in self.compatible_classes:
                layer.init_logvar(*args, **kwargs)

    def init_mu(self, *args, **kwargs):
        for name, layer in self.named_modules():
            if layer.__class__ in self.compatible_classes:
                layer.init_mu(*args, **kwargs)

    def project_logvar(self, prior, a=2):
        for (name, layer), (prior_name, prior_layer) in zip(self.named_modules(), prior.named_modules()):
            if layer.__class__ in self.compatible_classes:
                layer.project_logvar(prior_layer, a=a)

    def calc_kl_div(self, prior, device=None):
        if device is not None:
            kl_div = torch.tensor(0., dtype=torch.float).to(device)
        else:
            kl_div = torch.tensor(0., dtype=torch.float)

        for (name, layer), (prior_name, prior_layer) in zip(self.named_modules(), prior.named_modules()):
            if layer.__class__ in self.compatible_classes:
                kl_div += layer.calc_kl_div(prior_layer)

        return kl_div

class InflationModel(StochasticModel):
    def __init__(self, weight_size=1):
        super(InflationModel, self).__init__()

        # stochastic layer
        self.layer = StochasticLayer(weights_size=weight_size)

    def forward(self, x):
        # get output
        output = self.layer(x)
        return output


