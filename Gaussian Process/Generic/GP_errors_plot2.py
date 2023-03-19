import torch
import math
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
import matplotlib.pyplot as plt

# Define the Gaussian process regression model
class GPRegressionModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = RBFKernel(lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(0, 1))
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
# Set up the training data
train_x = torch.linspace(0, 1, 100)
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2
y_err = torch.abs(torch.randn(train_x.size()) * 0.1)
train_y = train_y + y_err

# Set up the Gaussian likelihood and GP model
likelihood = GaussianLikelihood()
model = GPRegressionModel(train_x, train_y, likelihood)

# Train the model
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iterations = 50
for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()

# Set up the test data
test_x = torch.linspace(0, 1, 51)
test_y = torch.sin(test_x * (2 * math.pi))

# Evaluate the model and print the test loss
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x))
    test_loss = ((observed_pred.mean - test_y)**2 / observed_pred.variance).mean()
    print('Test Loss: {:.3f}'.format(test_loss.item()))
    
    # Plot the results
    with gpytorch.settings.use_toeplitz(False):
        f, ax = plt.subplots()
        ax.errorbar(train_x.numpy(), train_y.numpy(), yerr=y_err.numpy(), fmt='k.', label='Observed Data')
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b', label='Mean')
        ax.fill_between(test_x.numpy(), 
                        observed_pred.mean.numpy() - 1.96 * observed_pred.variance.sqrt().numpy(),
                        observed_pred.mean.numpy() + 1.96 * observed_pred.variance.sqrt().numpy(), 
                        alpha=0.2, color='b')
        ax.set_ylim([-3, 3])
        ax.legend()
        plt.show()

