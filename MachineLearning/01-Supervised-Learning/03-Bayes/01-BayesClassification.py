import numpy as np

features = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 2, 1],
    [0, 0, 1],
    [2, 2, 0],
    [1, 1, 0],
    [0, 2, 1],
    [2, 0, 0],
    [2, 1, 0],
    [1, 0, 0]
])


class BayesModel:
    def __init__(self, prior=None, likelihood=None):
        self.prior = prior
        self.likelihood = likelihood
        self.posterior = None
        self.marginal = None

        self.features_posterior = None

    def fit(self, x, y, prior=None, likelihood=None):
        if prior is not None:
            self.prior = prior
        else:
            tmp = np.bincount(y)
            self.prior = tmp / tmp.sum()

        if likelihood is not None:
            self.likelihood = likelihood
        else:
            len_x = x.shape[0]
            len_y = len(np.bincount(y))
            self.likelihood = np.zeros((len_x, len_y))

            for i, item in enumerate(x):
                indices = np.where((x == item).all(axis=1))[0]
                self.likelihood[i, 0] = sum(1 - y[indices]) / sum(1 - y)
                self.likelihood[i, 1] = sum(y[indices]) / sum(y)

            prior_mul_likelihood = self.likelihood * self.prior
            self.marginal = np.sum(prior_mul_likelihood, axis=1)

            self.posterior = prior_mul_likelihood / self.marginal.reshape(-1, 1)


if __name__ == '__main__':
    model = BayesModel()
    model.fit(features[:, :-1], features[:, -1])
    posterior = model.posterior
    print(posterior)
