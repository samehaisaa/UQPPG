"""
Visualization utilities for Bayesian Neural Networks
Used to generate uncertainty visualizations for FactorizePhys model
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.signal import welch
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter, MaxNLocator


def plot_confidence_intervals(predictions, labels, fs, output_dir, filename_id, alpha=0.05):
    """Plot rPPG signal with confidence intervals
    
    Args:
        predictions: List of prediction samples from multiple Monte Carlo forward passes
        labels: Ground truth labels
        fs: Sampling frequency
        output_dir: Directory to save plots
        filename_id: Filename identifier
        alpha: Alpha value for confidence interval (default 0.05 for 95% CI)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Stack predictions to get mean and std
    stacked_predictions = np.stack(predictions)
    mean_prediction = np.mean(stacked_predictions, axis=0)
    std_prediction = np.std(stacked_predictions, axis=0)
    
    # Calculate confidence intervals
    z_score = 1.96  # 95% confidence interval
    lower_bound = mean_prediction - z_score * std_prediction
    upper_bound = mean_prediction + z_score * std_prediction
    
    # Time axis
    time = np.arange(len(mean_prediction)) / fs
    
    # Plot signal with confidence intervals
    plt.figure(figsize=(10, 6))
    plt.plot(time, labels, 'k-', alpha=0.7, label='Ground Truth')
    plt.plot(time, mean_prediction, 'b-', label='BNN Prediction')
    plt.fill_between(time, lower_bound, upper_bound, color='b', alpha=0.2, label='95% CI')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'rPPG Signal with BNN Confidence Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    output_path = os.path.join(output_dir, f'{filename_id}_confidence_intervals.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confidence interval plot to {output_path}")
    
    return output_path


def plot_weight_distributions(model, layer_names, output_dir, filename_id):
    """Plot weight distributions for specified Bayesian layers
    
    Args:
        model: BNN model
        layer_names: List of layer names to visualize
        output_dir: Directory to save plots
        filename_id: Filename identifier
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    paths = []
    
    # Find all Bayesian layers in the model
    bayesian_layers = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight_mu') and hasattr(module, 'weight_rho'):
            bayesian_layers[name] = module
    
    # If no specific layers provided, use all Bayesian layers
    if not layer_names:
        layer_names = list(bayesian_layers.keys())
    
    # Loop through requested layers
    for layer_name in layer_names:
        if layer_name not in bayesian_layers:
            continue
            
        layer = bayesian_layers[layer_name]
        
        # Calculate weight mean and std
        weight_mean = layer.weight_mu.flatten().detach().cpu().numpy()
        weight_std = torch.log1p(torch.exp(layer.weight_rho)).flatten().detach().cpu().numpy()
        
        # Create figure for weight distribution
        plt.figure(figsize=(10, 6))
        
        # Plot histogram of means
        plt.subplot(2, 1, 1)
        sns.histplot(weight_mean, kde=True)
        plt.title(f'Weight Mean Distribution - {layer_name}')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        
        # Plot histogram of standard deviations
        plt.subplot(2, 1, 2)
        sns.histplot(weight_std, kde=True)
        plt.title(f'Weight Std Distribution - {layer_name}')
        plt.xlabel('Standard Deviation')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'{filename_id}_weight_dist_{layer_name.replace(".", "_")}.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        paths.append(output_path)
        
    print(f"Saved {len(paths)} weight distribution plots to {output_dir}")
    
    return paths


def plot_uncertainty_heatmap(activations, uncertainties, output_dir, filename_id):
    """Plot heatmap of uncertainty in feature maps
    
    Args:
        activations: List of activation maps from different layers
        uncertainties: List of uncertainty maps from different layers
        output_dir: Directory to save plots
        filename_id: Filename identifier
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    paths = []
    
    # Create a custom colormap that goes from blue to red
    cmap = LinearSegmentedColormap.from_list('uncertainty', ['blue', 'red'])
    
    # Loop through each layer's activations and uncertainties
    for layer_idx, (activation, uncertainty) in enumerate(zip(activations, uncertainties)):
        # For 3D feature maps, take the mean across the channel dimension
        if len(activation.shape) > 3:
            activation = np.mean(activation, axis=1)
            uncertainty = np.mean(uncertainty, axis=1)
        
        # If still more than 2D, take the middle slice
        if len(activation.shape) > 2:
            mid_idx = activation.shape[1] // 2
            activation = activation[:, mid_idx, :]
            uncertainty = uncertainty[:, mid_idx, :]
        
        # Create figure
        plt.figure(figsize=(12, 5))
        
        # Plot activation
        plt.subplot(1, 2, 1)
        plt.imshow(activation, cmap='viridis')
        plt.title(f'Layer {layer_idx+1} Activation')
        plt.colorbar()
        
        # Plot uncertainty
        plt.subplot(1, 2, 2)
        plt.imshow(uncertainty, cmap=cmap)
        plt.title(f'Layer {layer_idx+1} Uncertainty')
        plt.colorbar()
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'{filename_id}_uncertainty_heatmap_layer{layer_idx+1}.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        paths.append(output_path)
    
    print(f"Saved {len(paths)} uncertainty heatmap plots to {output_dir}")
    
    return paths


def plot_bnn_vs_deterministic(bnn_predictions, det_predictions, labels, fs, output_dir, filename_id):
    """Plot comparison between BNN and deterministic model predictions
    
    Args:
        bnn_predictions: Predictions from BNN (with multiple samples)
        det_predictions: Predictions from deterministic model
        labels: Ground truth labels
        fs: Sampling frequency
        output_dir: Directory to save plots
        filename_id: Filename identifier
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Stack BNN predictions to get mean
    stacked_bnn = np.stack(bnn_predictions)
    mean_bnn = np.mean(stacked_bnn, axis=0)
    std_bnn = np.std(stacked_bnn, axis=0)
    
    # Time axis
    time = np.arange(len(mean_bnn)) / fs
    
    # Plot comparison of signals
    plt.figure(figsize=(10, 6))
    plt.plot(time, labels, 'k-', alpha=0.7, label='Ground Truth')
    plt.plot(time, det_predictions, 'g-', label='Deterministic Model')
    plt.plot(time, mean_bnn, 'b-', label='BNN Model')
    plt.fill_between(time, mean_bnn - std_bnn, mean_bnn + std_bnn, color='b', alpha=0.2, label='BNN Std')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Comparison of BNN vs Deterministic Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    output_path = os.path.join(output_dir, f'{filename_id}_bnn_vs_deterministic.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot frequency spectrum comparison
    plt.figure(figsize=(10, 6))
    
    # Calculate PSD for ground truth
    f_gt, psd_gt = welch(labels, fs=fs, nperseg=len(labels)//2)
    mask = (f_gt >= 0.6) & (f_gt <= 3.5)  # 36-210 BPM range
    f_gt = f_gt[mask]
    psd_gt = psd_gt[mask]
    
    # Calculate PSD for deterministic model
    f_det, psd_det = welch(det_predictions, fs=fs, nperseg=len(det_predictions)//2)
    f_det = f_det[mask]
    psd_det = psd_det[mask]
    
    # Calculate PSD for BNN model
    f_bnn, psd_bnn = welch(mean_bnn, fs=fs, nperseg=len(mean_bnn)//2)
    f_bnn = f_bnn[mask]
    psd_bnn = psd_bnn[mask]
    
    # Calculate PSD for each BNN sample and get confidence intervals
    psd_samples = []
    for sample in bnn_predictions:
        _, psd_sample = welch(sample, fs=fs, nperseg=len(sample)//2)
        psd_samples.append(psd_sample[mask])
    
    psd_samples = np.stack(psd_samples)
    psd_std = np.std(psd_samples, axis=0)
    
    # Plot PSDs
    plt.plot(f_gt, psd_gt, 'k-', alpha=0.7, label='Ground Truth')
    plt.plot(f_det, psd_det, 'g-', label='Deterministic Model')
    plt.plot(f_bnn, psd_bnn, 'b-', label='BNN Model')
    plt.fill_between(f_bnn, psd_bnn - psd_std, psd_bnn + psd_std, color='b', alpha=0.2, label='BNN Std')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.title(f'Power Spectral Density Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    output_path_psd = os.path.join(output_dir, f'{filename_id}_psd_comparison.pdf')
    plt.savefig(output_path_psd, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved BNN vs deterministic comparison plots to {output_dir}")
    
    return [output_path, output_path_psd]


def plot_hr_estimation_with_uncertainty(predictions, labels, fs, output_dir, filename_id):
    """Plot heart rate estimation with uncertainty
    
    Args:
        predictions: List of prediction samples from multiple Monte Carlo forward passes
        labels: Ground truth labels
        fs: Sampling frequency
        output_dir: Directory to save plots
        filename_id: Filename identifier
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Time window for HR calculation (in seconds)
    window_size = 10
    overlap = 5
    samples_per_window = window_size * fs
    step = (window_size - overlap) * fs
    
    # Time points for HR estimates
    time_points = []
    
    # Lists to store results
    gt_hrs = []
    pred_hrs = []
    hr_stds = []
    
    # Process ground truth
    for i in range(0, len(labels) - samples_per_window, step):
        # Window the signal
        window = labels[i:i+samples_per_window]
        
        # Calculate power spectrum
        f, psd = welch(window, fs=fs, nperseg=len(window))
        
        # Find the frequency with maximum power in the HR range (0.6-3.5 Hz, 36-210 BPM)
        mask = (f >= 0.6) & (f <= 3.5)
        if not any(mask):
            continue
            
        max_idx = np.argmax(psd[mask])
        hr_freq = f[mask][max_idx]
        hr_bpm = hr_freq * 60
        
        gt_hrs.append(hr_bpm)
        time_points.append((i + samples_per_window/2) / fs)
    
    # Process predictions
    for i in range(0, len(predictions[0]) - samples_per_window, step):
        # Get HR estimates from all prediction samples for this window
        hr_estimates = []
        
        for pred in predictions:
            window = pred[i:i+samples_per_window]
            
            # Calculate power spectrum
            f, psd = welch(window, fs=fs, nperseg=len(window))
            
            # Find the frequency with maximum power in the HR range
            mask = (f >= 0.6) & (f <= 3.5)
            if not any(mask):
                continue
                
            max_idx = np.argmax(psd[mask])
            hr_freq = f[mask][max_idx]
            hr_bpm = hr_freq * 60
            
            hr_estimates.append(hr_bpm)
        
        if hr_estimates:
            pred_hrs.append(np.mean(hr_estimates))
            hr_stds.append(np.std(hr_estimates))
    
    # Convert to numpy arrays
    time_points = np.array(time_points)
    gt_hrs = np.array(gt_hrs)
    pred_hrs = np.array(pred_hrs)
    hr_stds = np.array(hr_stds)
    
    # Plot HR with uncertainty
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, gt_hrs, 'k-', label='Ground Truth HR')
    plt.plot(time_points[:len(pred_hrs)], pred_hrs, 'b-', label='Estimated HR')
    plt.fill_between(time_points[:len(pred_hrs)], 
                     pred_hrs - 1.96 * hr_stds, 
                     pred_hrs + 1.96 * hr_stds, 
                     color='b', alpha=0.2, label='95% CI')
    plt.xlabel('Time (s)')
    plt.ylabel('Heart Rate (BPM)')
    plt.title('Heart Rate Estimation with Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    output_path = os.path.join(output_dir, f'{filename_id}_hr_uncertainty.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved HR estimation with uncertainty plot to {output_path}")
    
    return output_path 