"""
FactorizePhys: Matrix Factorization for Multidimensional Attention in Remote Physiological Sensing
NeurIPS 2024
Jitesh Joshi, Sos S. Agaian, and Youngjun Cho

Modified with Bayesian Neural Network capabilities for uncertainty quantification
"""

import torch
import torch.nn as nn
from neural_methods.model.FactorizePhys.FSAM import FeaturesFactorizationModule
from neural_methods.model.FactorizePhys.BayesianLayers import BayesianConvBlock3D, gather_kl_divergence

nf = [8, 12, 16]

model_config = {
    "MD_FSAM": True,
    "MD_TYPE": "NMF",
    "MD_TRANSFORM": "T_KAB",
    "MD_R": 1,
    "MD_S": 1,
    "MD_STEPS": 4,
    "MD_INFERENCE": False,
    "MD_RESIDUAL": False,
    "INV_T": 1,
    "ETA": 0.9,
    "RAND_INIT": True,
    "in_channels": 3,
    "data_channels": 4,
    "align_channels": nf[2] // 2,
    "height": 72,
    "weight": 72,
    "batch_size": 4,
    "frames": 160,
    "debug": False,
    "assess_latency": False,
    "num_trials": 20,
    "visualize": False,
    "ckpt_path": "",
    "data_path": "",
    "label_path": "",
    "enable_bnn": False,
    "bnn_prior_sigma_1": 1.0,
    "bnn_prior_sigma_2": 0.002,
    "bnn_prior_pi": 0.5,
    "bnn_kl_weight": 1e-6
}


class ConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, enable_bnn=False, 
                 prior_sigma_1=1.0, prior_sigma_2=0.002, prior_pi=0.5):
        super(ConvBlock3D, self).__init__()
        
        if enable_bnn:
            # Use Bayesian convolutional block
            self.conv_block_3d = BayesianConvBlock3D(
                in_channel, out_channel, kernel_size, stride, padding,
                prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi,
                enable_bnn=True
            )
        else:
            # Use standard convolutional block (original implementation)
            self.conv_block_3d = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding=padding, bias=False),
                nn.Tanh(),
                nn.InstanceNorm3d(out_channel),
            )

    def forward(self, x):
        return self.conv_block_3d(x)
    
    def deterministic_forward(self, x):
        """Deterministic forward pass for BNN or standard forward for non-BNN"""
        if hasattr(self.conv_block_3d, 'deterministic_forward'):
            return self.conv_block_3d.deterministic_forward(x)
        else:
            return self.conv_block_3d(x)


class rPPG_FeatureExtractor(nn.Module):
    def __init__(self, inCh, dropout_rate=0.1, debug=False, enable_bnn=False,
                 prior_sigma_1=1.0, prior_sigma_2=0.002, prior_pi=0.5):
        super(rPPG_FeatureExtractor, self).__init__()
        # inCh, out_channel, kernel_size, stride, padding

        self.debug = debug
        self.enable_bnn = enable_bnn
        #                                                        Input: #B, inCh, 160, 72, 72
        
        # Create feature extractor layers
        self.layer1 = ConvBlock3D(inCh, nf[0], [3, 3, 3], [1, 1, 1], [1, 1, 1], 
                                enable_bnn=enable_bnn, prior_sigma_1=prior_sigma_1, 
                                prior_sigma_2=prior_sigma_2, prior_pi=prior_pi)  #B, nf[0], 160, 72, 72
        
        self.layer2 = ConvBlock3D(nf[0], nf[1], [3, 3, 3], [1, 2, 2], [1, 0, 0], 
                                enable_bnn=enable_bnn, prior_sigma_1=prior_sigma_1, 
                                prior_sigma_2=prior_sigma_2, prior_pi=prior_pi) #B, nf[1], 160, 35, 35
        
        self.layer3 = ConvBlock3D(nf[1], nf[1], [3, 3, 3], [1, 1, 1], [1, 0, 0], 
                                enable_bnn=enable_bnn, prior_sigma_1=prior_sigma_1, 
                                prior_sigma_2=prior_sigma_2, prior_pi=prior_pi) #B, nf[1], 160, 33, 33
        
        self.dropout1 = nn.Dropout3d(p=dropout_rate)

        self.layer4 = ConvBlock3D(nf[1], nf[1], [3, 3, 3], [1, 1, 1], [1, 0, 0], 
                                enable_bnn=enable_bnn, prior_sigma_1=prior_sigma_1, 
                                prior_sigma_2=prior_sigma_2, prior_pi=prior_pi) #B, nf[1], 160, 31, 31
        
        self.layer5 = ConvBlock3D(nf[1], nf[2], [3, 3, 3], [1, 2, 2], [1, 0, 0], 
                                enable_bnn=enable_bnn, prior_sigma_1=prior_sigma_1, 
                                prior_sigma_2=prior_sigma_2, prior_pi=prior_pi) #B, nf[2], 160, 15, 15
        
        self.layer6 = ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], 
                                enable_bnn=enable_bnn, prior_sigma_1=prior_sigma_1, 
                                prior_sigma_2=prior_sigma_2, prior_pi=prior_pi) #B, nf[2], 160, 13, 13
        
        self.dropout2 = nn.Dropout3d(p=dropout_rate)
        
        # For storing intermediate activations
        self.activations = []
        self.uncertainties = []

    def forward(self, x):
        self.activations = []
        self.uncertainties = []
        
        # Forward through all layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dropout1(x)
        
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        voxel_embeddings = self.dropout2(x)
        
        if self.debug:
            print("rPPG Feature Extractor")
            print("     voxel_embeddings.shape", voxel_embeddings.shape)
        return voxel_embeddings
    
    def deterministic_forward(self, x):
        """Deterministic forward pass using mean weights only"""
        x = self.layer1.deterministic_forward(x)
        x = self.layer2.deterministic_forward(x)
        x = self.layer3.deterministic_forward(x)
        x = self.dropout1(x)
        
        x = self.layer4.deterministic_forward(x)
        x = self.layer5.deterministic_forward(x)
        x = self.layer6.deterministic_forward(x)
        voxel_embeddings = self.dropout2(x)
        
        return voxel_embeddings


class BVP_Head(nn.Module):
    def __init__(self, md_config, device, dropout_rate=0.1, debug=False, enable_bnn=False,
                prior_sigma_1=1.0, prior_sigma_2=0.002, prior_pi=0.5):
        super(BVP_Head, self).__init__()
        self.debug = debug
        self.enable_bnn = enable_bnn

        self.use_fsam = md_config["MD_FSAM"]
        self.md_type = md_config["MD_TYPE"]
        self.md_infer = md_config["MD_INFERENCE"]
        self.md_res = md_config["MD_RESIDUAL"]
        
        # Create conv block layers
        self.layer1 = ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], 
                                enable_bnn=enable_bnn, prior_sigma_1=prior_sigma_1, 
                                prior_sigma_2=prior_sigma_2, prior_pi=prior_pi) #B, nf[2], 160, 11, 11
        
        self.layer2 = ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], 
                                enable_bnn=enable_bnn, prior_sigma_1=prior_sigma_1, 
                                prior_sigma_2=prior_sigma_2, prior_pi=prior_pi) #B, nf[2], 160, 9, 9
        
        self.layer3 = ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], 
                                enable_bnn=enable_bnn, prior_sigma_1=prior_sigma_1, 
                                prior_sigma_2=prior_sigma_2, prior_pi=prior_pi) #B, nf[2], 160, 7, 7
        
        self.dropout = nn.Dropout3d(p=dropout_rate)

        if self.use_fsam:
            inC = nf[2]
            self.fsam = FeaturesFactorizationModule(inC, device, md_config, dim="3D", debug=debug)
            self.fsam_norm = nn.InstanceNorm3d(inC)
            self.bias1 = nn.Parameter(torch.tensor(1.0), requires_grad=True).to(device)
        else:
            inC = nf[2]

        # Create final layer components
        self.final_layer1 = ConvBlock3D(inC, nf[1], [3, 3, 3], [1, 1, 1], [1, 0, 0], 
                                     enable_bnn=enable_bnn, prior_sigma_1=prior_sigma_1, 
                                     prior_sigma_2=prior_sigma_2, prior_pi=prior_pi) #B, nf[1], 160, 5, 5
        
        self.final_layer2 = ConvBlock3D(nf[1], nf[0], [3, 3, 3], [1, 1, 1], [1, 0, 0], 
                                     enable_bnn=enable_bnn, prior_sigma_1=prior_sigma_1, 
                                     prior_sigma_2=prior_sigma_2, prior_pi=prior_pi) #B, nf[0], 160, 3, 3
        
        # The last layer is always deterministic to ensure stable output
        self.final_conv = nn.Conv3d(nf[0], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        
        # For storing uncertainties
        self.activations = []
        self.uncertainties = []

    def forward(self, voxel_embeddings, batch, length):
        self.activations = []
        self.uncertainties = []

        if self.debug:
            print("BVP Head")
            print("     voxel_embeddings.shape", voxel_embeddings.shape)

        # Process through conv block
        x = self.layer1(voxel_embeddings)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dropout(x)

        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            if "NMF" in self.md_type:
                att_mask, appx_error = self.fsam(x - x.min()) # to make it positive (>= 0)
            else:
                att_mask, appx_error = self.fsam(x)

            if self.debug:
                print("att_mask.shape", att_mask.shape)

            if self.md_res:
                # Multiplication with Residual connection
                factorized = torch.mul(x - x.min() + self.bias1, att_mask - att_mask.min() + self.bias1)
                factorized_embeddings = self.fsam_norm(factorized)
                factorized_embeddings = x + factorized_embeddings
            else:
                # Multiplication
                factorized = torch.mul(x - x.min() + self.bias1, att_mask - att_mask.min() + self.bias1)
                factorized_embeddings = self.fsam_norm(factorized)            

            # Process through final layers
            x = self.final_layer1(factorized_embeddings)
            x = self.final_layer2(x)
            x = self.final_conv(x)
        
        else:
            # Process through final layers
            x = self.final_layer1(x)
            x = self.final_layer2(x)
            x = self.final_conv(x)

        rPPG = x.view(-1, length)

        if self.debug:
            print("     rPPG.shape", rPPG.shape)
        
        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            return rPPG, factorized_embeddings, appx_error
        else:
            return rPPG
    
    def deterministic_forward(self, voxel_embeddings, batch, length):
        """Deterministic forward pass using mean weights only"""
        # Process through conv block with deterministic weights
        x = self.layer1.deterministic_forward(voxel_embeddings)
        x = self.layer2.deterministic_forward(x)
        x = self.layer3.deterministic_forward(x)
        x = self.dropout(x)

        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            if "NMF" in self.md_type:
                att_mask, appx_error = self.fsam(x - x.min())
            else:
                att_mask, appx_error = self.fsam(x)

            if self.md_res:
                factorized = torch.mul(x - x.min() + self.bias1, att_mask - att_mask.min() + self.bias1)
                factorized_embeddings = self.fsam_norm(factorized)
                factorized_embeddings = x + factorized_embeddings
            else:
                factorized = torch.mul(x - x.min() + self.bias1, att_mask - att_mask.min() + self.bias1)
                factorized_embeddings = self.fsam_norm(factorized)            

            # Process through final layers deterministically
            x = self.final_layer1.deterministic_forward(factorized_embeddings)
            x = self.final_layer2.deterministic_forward(x)
            x = self.final_conv(x)
        
        else:
            # Process through final layers deterministically
            x = self.final_layer1.deterministic_forward(x)
            x = self.final_layer2.deterministic_forward(x)
            x = self.final_conv(x)

        rPPG = x.view(-1, length)
        
        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            return rPPG, factorized_embeddings, appx_error
        else:
            return rPPG


class FactorizePhys(nn.Module):
    def __init__(self, frames, md_config, in_channels=3, dropout=0.1, device=torch.device("cpu"), debug=False):
        super(FactorizePhys, self).__init__()
        self.debug = debug

        self.in_channels = in_channels
        if self.in_channels == 1 or self.in_channels == 3:
            self.norm = nn.InstanceNorm3d(self.in_channels)
        elif self.in_channels == 4:
            self.rgb_norm = nn.InstanceNorm3d(3)
            self.thermal_norm = nn.InstanceNorm3d(1)
        else:
            print("Unsupported input channels")
        
        # Apply default values for any missing parameters
        for key in model_config:
            if key not in md_config:
                md_config[key] = model_config[key]
        
        # Extract BNN parameters
        self.enable_bnn = md_config.get("enable_bnn", False)
        self.prior_sigma_1 = md_config.get("bnn_prior_sigma_1", 1.0)
        self.prior_sigma_2 = md_config.get("bnn_prior_sigma_2", 0.002)
        self.prior_pi = md_config.get("bnn_prior_pi", 0.5)
        self.kl_weight = md_config.get("bnn_kl_weight", 1e-6)
        
        # Keep other parameters
        self.use_fsam = md_config["MD_FSAM"]
        self.md_infer = md_config["MD_INFERENCE"]

        if self.debug:
            print("nf:", nf)
            if self.enable_bnn:
                print("BNN enabled with prior_sigma_1:", self.prior_sigma_1)
                print("prior_sigma_2:", self.prior_sigma_2)
                print("prior_pi:", self.prior_pi)
                print("kl_weight:", self.kl_weight)

        # Create model components
        self.rppg_feature_extractor = rPPG_FeatureExtractor(
            self.in_channels, 
            dropout_rate=dropout, 
            debug=debug,
            enable_bnn=self.enable_bnn,
            prior_sigma_1=self.prior_sigma_1,
            prior_sigma_2=self.prior_sigma_2,
            prior_pi=self.prior_pi
        )

        self.rppg_head = BVP_Head(
            md_config, 
            device=device, 
            dropout_rate=dropout, 
            debug=debug,
            enable_bnn=self.enable_bnn,
            prior_sigma_1=self.prior_sigma_1,
            prior_sigma_2=self.prior_sigma_2,
            prior_pi=self.prior_pi
        )
        
    def forward(self, x): # [batch, Features=3, Temp=frames, Width=32, Height=32]
        [batch, channel, length, width, height] = x.shape
        
        # Apply normalization based on input channels
        if self.in_channels == 1:
            x = self.norm(x)
        elif self.in_channels == 3:
            x = self.norm(x)
        elif self.in_channels == 4:
            rgb = self.rgb_norm(x[:, 0:3, ...])
            thermal = self.thermal_norm(x[:, 3:4, ...])
            x = torch.cat([rgb, thermal], dim=1)
        
        # Forward through feature extractor
        voxel_embeddings = self.rppg_feature_extractor(x)
        
        # Forward through BVP head
        if (self.md_infer or self.training or self.debug) and self.use_fsam:
            rPPG, factorized_embeddings, appx_error = self.rppg_head(voxel_embeddings, batch, length)
            return rPPG, factorized_embeddings, appx_error
        else:
            rPPG = self.rppg_head(voxel_embeddings, batch, length)
            return rPPG
    
    def deterministic_forward(self, x):
        """Deterministic forward pass using mean weights only"""
        [batch, channel, length, width, height] = x.shape
        
        # Apply normalization based on input channels
        if self.in_channels == 1:
            x = self.norm(x)
        elif self.in_channels == 3:
            x = self.norm(x)
        elif self.in_channels == 4:
            rgb = self.rgb_norm(x[:, 0:3, ...])
            thermal = self.thermal_norm(x[:, 3:4, ...])
            x = torch.cat([rgb, thermal], dim=1)
        
        # Forward through feature extractor deterministically
        voxel_embeddings = self.rppg_feature_extractor.deterministic_forward(x)
        
        # Forward through BVP head deterministically
        if (self.md_infer or not self.training or self.debug) and self.use_fsam:
            rPPG, factorized_embeddings, appx_error = self.rppg_head.deterministic_forward(voxel_embeddings, batch, length)
            return rPPG, factorized_embeddings, appx_error
        else:
            rPPG = self.rppg_head.deterministic_forward(voxel_embeddings, batch, length)
            return rPPG
    
    def monte_carlo_forward(self, x, n_samples=10):
        """Run multiple forward passes for uncertainty estimation"""
        samples = []
        [batch, channel, length, width, height] = x.shape
        
        for _ in range(n_samples):
            if self.use_fsam:
                rPPG, _, _ = self.forward(x)
            else:
                rPPG = self.forward(x)
                
            # Ensure predictions match the input temporal length
            if rPPG.shape[1] > length:
                rPPG = rPPG[:, :length]
                
            samples.append(rPPG.detach().cpu())
        return samples
    
    def kl_divergence(self):
        """Calculate total KL divergence for all Bayesian layers"""
        if not self.enable_bnn:
            return torch.tensor(0.0).to(next(self.parameters()).device)
        
        return gather_kl_divergence(self)
