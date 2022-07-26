import matplotlib.pyplot as plt
import numpy as np
import torch
from numba import jit
from scipy.optimize import minimize
from torch.nn import functional as F
from tqdm import tqdm


@jit(nopython=True)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# This is an implementation of Algorithm 3 (Regularised Bayesian Logistic Regression with a Laplace Approximation)
# from "An Empirical Evaluation of Thompson Sampling" by Olivier Chapelle & Lihong Li
# https://proceedings.neurips.cc/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf

class PyTorchLogisticRegression(torch.nn.Module):
    def __init__(self, n_dim, n_items):
        super(PyTorchLogisticRegression, self).__init__()
        self.m = torch.nn.Parameter(torch.Tensor(n_items, n_dim + 1))
        torch.nn.init.normal_(self.m, mean=0.0, std=1.0)
        self.prev_iter_m = self.m.detach().clone()
        self.q = torch.ones((n_items, n_dim + 1))
        self.logloss = torch.nn.BCELoss(reduction='sum')
        self.eval()

    def forward(self, x, sample=False):
        ''' Predict outcome for all items, allow for posterior sampling '''
        if sample:
            return torch.sigmoid(F.linear(x, self.m + torch.normal(mean=0.0, std=1.0/torch.sqrt(self.q))))
        else:
            return torch.sigmoid(F.linear(x, self.m))

    def predict_item(self, x, a):
        ''' Predict outcome for an item a, only MAP '''
        return torch.sigmoid((x * self.m[a]).sum(axis=1))

    def loss(self, predictions, labels):
        prior_dist = self.q[:, :-1] * (self.prev_iter_m[:, :-1] - self.m[:, :-1])**2
        return 0.5 * prior_dist.sum() + self.logloss(predictions, labels)

    def laplace_approx(self, X, item):
        P = (1 + torch.exp(1 - X.matmul(self.m[item, :].T))) ** (-1)
        self.q[item, :] += (P*(1-P)).T.matmul(X ** 2).squeeze(0)

    def update_prior(self):
        self.prev_iter_m = self.m.detach().clone()


class PyTorchWinRateEstimator(torch.nn.Module):
    def __init__(self):
        super(PyTorchWinRateEstimator, self).__init__()
        # Input  P(click), the value, and the bid shading factor
        self.model = torch.nn.Sequential(
            torch.nn.Linear(3, 1, bias=True),
            torch.nn.Sigmoid()
        )
        self.eval()

    def forward(self, x):
        return self.model(x)


class BidShadingPolicy(torch.nn.Module):
    def __init__(self):
        super(BidShadingPolicy, self).__init__()
        # Input: P(click), value
        # Output: mu, sigma for Gaussian bid shading distribution
        # Learnt to maximise E[P(win|gamma)*(value - price)] when gamma ~ N(mu, sigma)
        self.shared_linear = torch.nn.Linear(2, 2, bias=True)

        self.mu_linear_hidden = torch.nn.Linear(2, 2)
        self.mu_linear_out = torch.nn.Linear(2, 1)

        self.sigma_linear_hidden = torch.nn.Linear(2, 2)
        self.sigma_linear_out = torch.nn.Linear(2, 1)
        self.eval()

        self.min_sigma = 1e-2

    def forward(self, x):
        x = self.shared_linear(x)
        mu = torch.nn.Softplus()(self.mu_linear_out(torch.nn.Softplus()(x)))
        sigma = torch.nn.Softplus()(self.sigma_linear_out(torch.nn.Softplus()(x))) + self.min_sigma
        dist = torch.distributions.normal.Normal(mu, sigma)
        sampled_value = dist.rsample()
        propensity = torch.exp(dist.log_prob(sampled_value))
        sampled_value = torch.clip(sampled_value, min=0.0, max=1.0)
        return sampled_value, propensity


class BidShadingContextualBandit(torch.nn.Module):
    def __init__(self, loss, winrate_model=None):
        super(BidShadingContextualBandit, self).__init__()

        self.shared_linear = torch.nn.Linear(2, 2, bias=True)

        self.mu_linear_out = torch.nn.Linear(2, 1)

        self.sigma_linear_out = torch.nn.Linear(2, 1)
        self.eval()

        self.min_sigma = 1e-2

        self.loss_name = loss

        self.model_initialised = False

    def initialise_policy(self, observed_contexts, observed_gammas):
        # The first time, train the policy to imitate the logging policy
        self.train()
        epochs = 8192 * 2
        lr = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)

        criterion = torch.nn.MSELoss()
        losses = []
        best_epoch, best_loss = -1, np.inf
        for epoch in tqdm(range(int(epochs)), desc=f'Initialising Policy'):
            optimizer.zero_grad()  # Setting our stored gradients equal to zero
            predicted_mu_gammas = torch.nn.Softplus()(self.mu_linear_out(torch.nn.Softplus()(self.shared_linear(observed_contexts))))
            predicted_sigma_gammas = torch.nn.Softplus()(self.sigma_linear_out(torch.nn.Softplus()(self.shared_linear(observed_contexts))))
            loss = criterion(predicted_mu_gammas.squeeze(), observed_gammas) + criterion(predicted_sigma_gammas.squeeze(), torch.ones_like(observed_gammas) * .05)
            loss.backward()  # Computes the gradient of the given tensor w.r.t. the weights/bias
            optimizer.step()  # Updates weights and biases with the optimizer (SGD)
            losses.append(loss.item())
            if (best_loss - losses[-1]) > 1e-6:
                best_epoch = epoch
                best_loss = losses[-1]
            elif epoch - best_epoch > 512:
                print(f'Stopping at Epoch {epoch}')
                break

        fig, ax = plt.subplots()
        plt.title(f'Initialising policy')
        plt.plot(losses, label=r'Loss')
        plt.ylabel('MSE with logging policy')
        plt.legend()
        fig.set_tight_layout(True)
        #plt.show()

        print('Predicted mu Gammas: ', predicted_mu_gammas.min(), predicted_mu_gammas.max(), predicted_mu_gammas.mean())
        print('Predicted sigma Gammas: ', predicted_sigma_gammas.min(), predicted_sigma_gammas.max(), predicted_sigma_gammas.mean())

    def forward(self, x):
        x = self.shared_linear(x)
        dist = torch.distributions.normal.Normal(
            torch.nn.Softplus()(self.mu_linear_out(torch.nn.Softplus()(x))),
            torch.nn.Softplus()(self.sigma_linear_out(torch.nn.Softplus()(x))) + self.min_sigma
        )
        sampled_value = dist.rsample()
        propensity = torch.exp(dist.log_prob(sampled_value))
        sampled_value = torch.clip(sampled_value, min=0.0, max=1.0)
        return sampled_value, propensity

    def normal_pdf(self, x, gamma):
        # Get distribution over bid shading factors
        x = self.shared_linear(x)
        mu = torch.nn.Softplus()(self.mu_linear_out(torch.nn.Softplus()(x)))
        sigma = torch.nn.Softplus()(self.sigma_linear_out(torch.nn.Softplus()(x))) + self.min_sigma
        mu = mu.squeeze()
        sigma = sigma.squeeze()
        # Compute the density for gamma under a Gaussian centered at mu -- prevent overflow
        return mu, sigma, torch.clip(torch.exp(-((mu - gamma) / sigma)**2/2) / (sigma * np.sqrt(2 * np.pi)), min=1e-30)

    def loss(self, observed_context, observed_gamma, logging_propensity, utility, utility_estimates=None, winrate_model=None, KL_weight=5e-2, importance_weight_clipping_eps=torch.inf):

        mean_gamma_target, sigma_gamma_target, target_propensities = self.normal_pdf(observed_context, observed_gamma)

        # If not initialised, do a single round of on-policy REINFORCE
        # The issue is that without proper initialisation, propensities vanish
        if (self.loss_name == 'REINFORCE'): # or (not self.model_initialised)
            return (-target_propensities * utility).mean()

        elif self.loss_name == 'REINFORCE_offpolicy':
            importance_weights = target_propensities / logging_propensity
            return (-importance_weights * utility).mean()

        elif self.loss_name == 'TRPO':
            # https://arxiv.org/abs/1502.05477
            importance_weights = target_propensities / logging_propensity
            expected_utility = torch.mean(importance_weights * utility)
            KLdiv = (sigma_gamma_target**2 + (mean_gamma_target - observed_gamma)**2) / (2 * sigma_gamma_target**2) - 0.5
            # Simpler proxy for KL divergence
            # KLdiv = (mean_gamma_target - observed_gamma)**2
            return - expected_utility + KLdiv.mean() * KL_weight

        elif self.loss_name == 'PPO':
            # https://arxiv.org/pdf/1707.06347.pdf
            # NOTE: clipping is actually proposed in an additive manner
            importance_weights = target_propensities / logging_propensity
            clipped_importance_weights = torch.clip(importance_weights,
                                                    min=1.0/importance_weight_clipping_eps,
                                                    max=importance_weight_clipping_eps)
            return - torch.min(importance_weights * utility, clipped_importance_weights * utility).mean()

        elif self.loss_name == 'Doubly Robust':
            importance_weights = target_propensities / logging_propensity

            DR_IPS = (utility - utility_estimates) * torch.clip(importance_weights, min=1.0/importance_weight_clipping_eps, max=importance_weight_clipping_eps)

            dist = torch.distributions.normal.Normal(
                mean_gamma_target,
                sigma_gamma_target
            )

            sampled_gamma = torch.clip(dist.rsample(), min=0.0, max=1.0)
            features_for_p_win = torch.hstack((observed_context, sampled_gamma.reshape(-1,1)))

            W = winrate_model(features_for_p_win).squeeze()

            V = observed_context[:,0].squeeze() * observed_context[:,1].squeeze()
            P = observed_context[:,0].squeeze() * observed_context[:,1].squeeze() * sampled_gamma

            DR_DM = W * (V - P)

            return -(DR_IPS + DR_DM).mean()
