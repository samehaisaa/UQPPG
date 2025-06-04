"""
Bayesian Neural Network Layers for FactorizePhys
Based on the Bayes by Backprop approach for uncertainty estimation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class ModuleWrapper(nn.Module):
    """Wrapper for nn.Module with support for non-deterministic forward pass"""
    def __init__(self):
        super(ModuleWrapper, self).__init__()

    def deterministic_forward(self, *args, **kwargs):
        """Forward pass with deterministic weights (mean only)"""
        pass


class BayesianConv3d(ModuleWrapper):
    """Bayesian 3D Convolutional layer with weight distributions"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, bias=True, prior_sigma_1=1.0, prior_sigma_2=0.002, 
                 prior_pi=0.5):
        super(BayesianConv3d, self).__init__()
        
        # Register layer parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        elif isinstance(kernel_size, list):
            self.kernel_size = tuple(kernel_size)
        else:
            self.kernel_size = (kernel_size,) * 3
        if isinstance(stride, tuple):
            self.stride = stride
        elif isinstance(stride, list):
            self.stride = tuple(stride)
        else:
            self.stride = (stride,) * 3
        if isinstance(padding, tuple):
            self.padding = padding
        elif isinstance(padding, list):
            self.padding = tuple(padding)
        else:
            self.padding = (padding,) * 3
        if isinstance(dilation, tuple):
            self.dilation = dilation
        elif isinstance(dilation, list):
            self.dilation = tuple(dilation)
        else:
            self.dilation = (dilation,) * 3
        self.groups = 1
        self.has_bias = bias
        
        # Compute weight dimensions
        self.weight_shape = (out_channels, in_channels, *self.kernel_size)
        self.weight_size = out_channels * in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        
        # Weight parameters - mean and rho (transformed variance parameter)
        self.weight_mu = Parameter(torch.Tensor(*self.weight_shape))
        self.weight_rho = Parameter(torch.Tensor(*self.weight_shape))
        
        # Bias parameters if applicable
        if self.has_bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_rho = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
            
        # Initialize priors
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        
        # Initialize weights and biases
        self.reset_parameters()
        
        # Other variables
        self.weight = None
        self.bias = None
        self.kl_divergence = 0.0
    
    def reset_parameters(self):
        """Initialize the weights and bias parameters"""
        # Weight initialization
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_rho.data.fill_(-3.0)  # Initialize to small variance
        
        if self.has_bias:
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_rho.data.fill_(-3.0)
    
    def forward(self, x):
        """Stochastic forward pass using sampled weights and biases"""
        # Sample weights using reparameterization trick
        # Weight: µ + log(1 + exp(ρ)) * ε where ε ~ N(0, 1)
        epsilon_w = torch.randn_like(self.weight_mu)
        sigma_w = torch.log1p(torch.exp(self.weight_rho))
        self.weight = self.weight_mu + sigma_w * epsilon_w
        
        # Sample bias if needed
        if self.has_bias:
            epsilon_b = torch.randn_like(self.bias_mu)
            sigma_b = torch.log1p(torch.exp(self.bias_rho))
            self.bias = self.bias_mu + sigma_b * epsilon_b
        else:
            self.bias = None
            
        # Calculate KL divergence between posterior and prior
        # Store for later use in loss computation
        self.kl_divergence = self._compute_kl_divergence()
        
        # Perform convolution operation
        output = F.conv3d(
            input=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        
        return output
    
    def deterministic_forward(self, x):
        """Deterministic forward pass using only mean values (no sampling)"""
        # Use only the mean for prediction without sampling
        return F.conv3d(
            input=x,
            weight=self.weight_mu,
            bias=self.bias_mu if self.has_bias else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
    
    def _compute_kl_divergence(self):
        """Compute KL divergence between posterior and prior for weights and biases"""
        # Weight KL divergence
        sigma_w = torch.log1p(torch.exp(self.weight_rho))
        kl_w = self._kl_normal_mixture(self.weight_mu, sigma_w)
        
        # Bias KL divergence if applicable
        if self.has_bias:
            sigma_b = torch.log1p(torch.exp(self.bias_rho))
            kl_b = self._kl_normal_mixture(self.bias_mu, sigma_b)
            return kl_w + kl_b
        
        return kl_w
    
    def _kl_normal_mixture(self, mu, sigma):
        """KL divergence between posterior N(μ,σ²) and prior mixture of Gaussians"""
        # log[q(w|θ)/p(w)]
        sigma2 = sigma ** 2
        log_sigma2 = torch.log(sigma2)
        
        # Prior: mixture of two Gaussians with different variances
        # p(w) = π * N(0, σ₁²) + (1-π) * N(0, σ₂²)
        prior_sigma2_1 = torch.tensor(self.prior_sigma_1 ** 2, device=mu.device)
        prior_sigma2_2 = torch.tensor(self.prior_sigma_2 ** 2, device=mu.device)
        prior_pi = torch.tensor(self.prior_pi, device=mu.device)
        
        # Calculate components for the KL divergence formula
        # Reshape prior parameters to ensure proper broadcasting
        prior_sigma2_1 = prior_sigma2_1.expand_as(sigma2)
        prior_sigma2_2 = prior_sigma2_2.expand_as(sigma2)
        prior_pi = prior_pi.expand_as(sigma2)
        
        kl_component_1 = torch.log(prior_pi / torch.sqrt(prior_sigma2_1) + 
                                  (1 - prior_pi) / torch.sqrt(prior_sigma2_2))
        kl_component_1 -= torch.log(1.0 / torch.sqrt(sigma2))
        
        kl_component_2 = 0.5 * (log_sigma2 + (mu ** 2 + sigma2) / prior_sigma2_1)
        kl_component_3 = 0.5 * (log_sigma2 + (mu ** 2 + sigma2) / prior_sigma2_2)
        
        # Logsumexp for numerical stability
        # We directly calculate log(π * exp(-0.5 * ((μ²+σ²)/σ₁² + log(σ₁²))) + (1-π) * exp(-0.5 * ((μ²+σ²)/σ₂² + log(σ₂²))))
        kl_component_mix = torch.logsumexp(
            torch.stack([
                torch.log(prior_pi) - kl_component_2,
                torch.log(1 - prior_pi) - kl_component_3
            ]),
            dim=0
        )
        
        kl = kl_component_1 - kl_component_mix
        return kl.sum()


class BayesianConvBlock3D(ModuleWrapper):
    """Bayesian 3D Convolutional block with activation and batch norm"""
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, 
                 prior_sigma_1=1.0, prior_sigma_2=0.002, prior_pi=0.5, enable_bnn=False):
        super(BayesianConvBlock3D, self).__init__()
        
        self.enable_bnn = enable_bnn
        
        if enable_bnn:
            # Bayesian convolution
            self.conv = BayesianConv3d(
                in_channel, out_channel, kernel_size, stride, padding=padding,
                prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi
            )
        else:
            # Standard convolution
            self.conv = nn.Conv3d(
                in_channel, out_channel, kernel_size, stride, padding=padding, bias=False
            )
        
        # Shared components
        self.activation = nn.Tanh()
        self.norm = nn.InstanceNorm3d(out_channel)
        
    def forward(self, x):
        """Forward pass with either Bayesian or standard conv"""
        x = self.conv(x)
        x = self.activation(x)
        x = self.norm(x)
        return x
    
    def deterministic_forward(self, x):
        """Deterministic forward pass"""
        if self.enable_bnn:
            x = self.conv.deterministic_forward(x)
        else:
            x = self.conv(x)
        x = self.activation(x)
        x = self.norm(x)
        return x
    
    @property
    def kl_divergence(self):
        """Return KL divergence if BNN is enabled"""
        if self.enable_bnn:
            return self.conv.kl_divergence
        return 0.0


class BayesianLinear(ModuleWrapper):
    """Bayesian Linear layer with weight distributions"""
    def __init__(self, in_features, out_features, bias=True, 
                 prior_sigma_1=1.0, prior_sigma_2=0.002, prior_pi=0.5):
        super(BayesianLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        
        # Weight parameters
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = Parameter(torch.Tensor(out_features, in_features))
        
        # Bias parameters if applicable
        if self.has_bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_rho = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
            
        # Initialize priors
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        
        # Initialize weights and biases
        self.reset_parameters()
        
        # Other variables
        self.weight = None
        self.bias = None
        self.kl_divergence = 0.0
    
    def reset_parameters(self):
        """Initialize the weights and bias parameters"""
        stdv = 1.0 / math.sqrt(self.in_features)
        
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_rho.data.fill_(-3.0)
        
        if self.has_bias:
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_rho.data.fill_(-3.0)
    
    def forward(self, x):
        """Stochastic forward pass using sampled weights and biases"""
        # Sample weights using reparameterization trick
        epsilon_w = torch.randn_like(self.weight_mu)
        sigma_w = torch.log1p(torch.exp(self.weight_rho))
        self.weight = self.weight_mu + sigma_w * epsilon_w
        
        # Sample bias if needed
        if self.has_bias:
            epsilon_b = torch.randn_like(self.bias_mu)
            sigma_b = torch.log1p(torch.exp(self.bias_rho))
            self.bias = self.bias_mu + sigma_b * epsilon_b
        else:
            self.bias = None
            
        # Calculate KL divergence
        self.kl_divergence = self._compute_kl_divergence()
        
        # Perform linear operation
        return F.linear(x, self.weight, self.bias)
    
    def deterministic_forward(self, x):
        """Deterministic forward pass using only mean values (no sampling)"""
        return F.linear(
            x, 
            self.weight_mu,
            self.bias_mu if self.has_bias else None
        )
    
    def _compute_kl_divergence(self):
        """Compute KL divergence between posterior and prior for weights and biases"""
        # Weight KL divergence
        sigma_w = torch.log1p(torch.exp(self.weight_rho))
        kl_w = self._kl_normal_mixture(self.weight_mu, sigma_w)
        
        # Bias KL divergence if applicable
        if self.has_bias:
            sigma_b = torch.log1p(torch.exp(self.bias_rho))
            kl_b = self._kl_normal_mixture(self.bias_mu, sigma_b)
            return kl_w + kl_b
        
        return kl_w
    
    def _kl_normal_mixture(self, mu, sigma):
        """KL divergence between posterior N(μ,σ²) and prior mixture of Gaussians"""
        # log[q(w|θ)/p(w)]
        sigma2 = sigma ** 2
        log_sigma2 = torch.log(sigma2)
        
        # Prior: mixture of two Gaussians with different variances
        # p(w) = π * N(0, σ₁²) + (1-π) * N(0, σ₂²)
        prior_sigma2_1 = torch.tensor(self.prior_sigma_1 ** 2, device=mu.device)
        prior_sigma2_2 = torch.tensor(self.prior_sigma_2 ** 2, device=mu.device)
        prior_pi = torch.tensor(self.prior_pi, device=mu.device)
        
        # Calculate components for the KL divergence formula
        # Reshape prior parameters to ensure proper broadcasting
        prior_sigma2_1 = prior_sigma2_1.expand_as(sigma2)
        prior_sigma2_2 = prior_sigma2_2.expand_as(sigma2)
        prior_pi = prior_pi.expand_as(sigma2)
        
        kl_component_1 = torch.log(prior_pi / torch.sqrt(prior_sigma2_1) + 
                                  (1 - prior_pi) / torch.sqrt(prior_sigma2_2))
        kl_component_1 -= torch.log(1.0 / torch.sqrt(sigma2))
        
        kl_component_2 = 0.5 * (log_sigma2 + (mu ** 2 + sigma2) / prior_sigma2_1)
        kl_component_3 = 0.5 * (log_sigma2 + (mu ** 2 + sigma2) / prior_sigma2_2)
        
        # Logsumexp for numerical stability
        # We directly calculate log(π * exp(-0.5 * ((μ²+σ²)/σ₁² + log(σ₁²))) + (1-π) * exp(-0.5 * ((μ²+σ²)/σ₂² + log(σ₂²))))
        kl_component_mix = torch.logsumexp(
            torch.stack([
                torch.log(prior_pi) - kl_component_2,
                torch.log(1 - prior_pi) - kl_component_3
            ]),
            dim=0
        )
        
        kl = kl_component_1 - kl_component_mix
        return kl.sum()


def gather_kl_divergence(model):
    """Gather KL divergence from all Bayesian layers in a model"""
    kl_sum = 0.0
    for module in model.modules():
        if hasattr(module, 'kl_divergence'):
            kl_sum += module.kl_divergence
    return kl_sum 
