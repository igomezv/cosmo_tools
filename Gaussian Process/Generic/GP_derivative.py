import torch
import gpytorch
import matplotlib.pyplot as plt

# Define the data
x = torch.linspace(0, 1, 100)
y = torch.sin(x * (2 * torch.Tensor([3.1415])))
dy = 2 * torch.cos(x * (2 * torch.Tensor([3.1415])) * 2 * torch.Tensor([3.1415]))

# Define the Gaussian Process model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.ProductKernel(
                gpytorch.kernels.MaternKernel(nu=1.5), gpytorch.kernels.RBFKernel()))
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Initialize the likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(x, dy, likelihood)

# Train the model
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
training_iterations = 50
for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(x)
    loss = -mll(output, dy)
    loss.backward()
    optimizer.step()

# Evaluate the model
model.eval()
likelihood.eval()
with torch.no_grad():
    test_x = torch.linspace(0, 1, 51)
    observed_pred = likelihood(model(test_x))
    mean = observed_pred.mean
    lower, upper = observed_pred.confidence_region()

# Plot the results
plt.plot(x.numpy(), dy.numpy(), 'k.')
plt.plot(test_x.numpy(), mean.numpy(), 'b')
plt.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
plt.xlabel('x')
plt.ylabel('dy/dx')
plt.show()
