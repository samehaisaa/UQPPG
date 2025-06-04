# Bayesian Neural Networks for rPPG Uncertainty Quantification

This extension to the [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) adds Bayesian Neural Network (BNN) capabilities to the FactorizePhys model, enabling uncertainty quantification for remote photoplethysmography.

## Features

- **Drop-in BNN Enhancement**: Use the standard FactorizePhys model with Bayesian capabilities
- **Uncertainty Quantification**: Generate confidence intervals for rPPG predictions
- **Weight Distribution Visualization**: Analyze learned parameter distributions
- **Uncertainty Heatmaps**: Visualize uncertainty in different layers
- **Heart Rate Uncertainty**: Estimate uncertainty in derived heart rate measurements

## Usage

### Method 1: Using a BNN Config File

The simplest way to use BNN features is with the provided config file:

```bash
python main.py --config_file configs/train_configs/UBFC-rPPG_PURE_FactorizePhys_FSAM_Res_BNN.yaml
```

### Method 2: Command-line Options

You can also enable BNN features on any existing FactorizePhys configuration:

```bash
python main.py --config_file configs/train_configs/UBFC-rPPG_PURE_FactorizePhys_FSAM_Res.yaml --enable_bnn
```

Additional BNN parameters can be customized:

```bash
python main.py --config_file configs/train_configs/UBFC-rPPG_PURE_FactorizePhys_FSAM_Res.yaml --enable_bnn --bnn_kl_weight 1e-5 --bnn_samples 20
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `enable_bnn` | Enable Bayesian Neural Network | `False` |
| `bnn_kl_weight` | Weight for KL divergence loss term | `1e-6` |
| `bnn_prior_sigma_1` | Prior sigma 1 (larger variance) | `1.0` |
| `bnn_prior_sigma_2` | Prior sigma 2 (smaller variance) | `0.002` |
| `bnn_prior_pi` | Prior mixture proportion | `0.5` |
| `bnn_samples` | Number of Monte Carlo samples for uncertainty | `10` |

## Visualizations

When testing a BNN model, the following visualizations are automatically generated:

1. **Confidence Interval Plots**: Show the rPPG signal with 95% confidence intervals
2. **BNN vs Deterministic Comparison**: Compare BNN with standard model performance
3. **Heart Rate Uncertainty**: Visualize uncertainty in heart rate estimates
4. **Weight Distributions**: Analyze learned parameter distributions

All visualizations are saved to the directory specified in the config's `TEST.OUTPUT_SAVE_DIR` parameter.

## Implementation Details

The BNN implementation uses a Bayes-by-Backprop approach with:

- **Weight Posteriors**: Parameterized by mean (μ) and variance (ρ)
- **Scale Mixture Prior**: Two Gaussian components to encourage both large and small weights
- **Reparameterization Trick**: For backpropagation through random sampling
- **KL Divergence Regularization**: Balances fit quality with model complexity

## Example Workflow

1. **Train a BNN Model**:
   ```bash
   python main.py --config_file configs/train_configs/UBFC-rPPG_PURE_FactorizePhys_FSAM_Res_BNN.yaml
   ```

2. **Test and Generate Visualizations**:
   ```bash
   python main.py --config_file configs/train_configs/UBFC-rPPG_PURE_FactorizePhys_FSAM_Res_BNN.yaml --toolbox_mode only_test --inference_model_path /path/to/saved/model.pth
   ```

3. **View Results**:
   Check the visualizations in the specified output directory.

## Citing

If you use this BNN extension in your research, please cite both the original rPPG-Toolbox and this extension:

```bibtex
@article{rppg-toolbox,
  title={rPPG-Toolbox: Deep Remote PPG Toolbox},
  author={...},
  journal={...},
  year={...}
}

@article{factorizephys-bnn,
  title={Uncertainty Quantification in Remote Photoplethysmography with Bayesian Neural Networks},
  author={...},
  journal={...},
  year={...}
}
``` 