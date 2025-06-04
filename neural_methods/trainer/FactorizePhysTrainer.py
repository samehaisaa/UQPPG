"""
FactorizePhys: Matrix Factorization for Multidimensional Attention in Remote Physiological Sensing
NeurIPS 2024
Jitesh Joshi, Sos S. Agaian, and Youngjun Cho

Modified with Bayesian Neural Network capabilities for uncertainty quantification
"""

import os
import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.FactorizePhys.FactorizePhys import FactorizePhys
from neural_methods.model.FactorizePhys.FactorizePhysBig import FactorizePhysBig
from neural_methods.model.FactorizePhys.UncertaintyVisualization import (
    plot_confidence_intervals, 
    plot_weight_distributions,
    plot_uncertainty_heatmap,
    plot_bnn_vs_deterministic,
    plot_hr_estimation_with_uncertainty
)
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm


class FactorizePhysTrainer(BaseTrainer):

    @staticmethod
    def add_trainer_args(parser):
        """Adds arguments to Parser for training process"""
        parser = BaseTrainer.add_trainer_args(parser)
        parser.add_argument('--enable_bnn', action='store_true', help='Enable Bayesian Neural Network')
        parser.add_argument('--bnn_kl_weight', default=1e-6, type=float, help='Weight for KL divergence loss')
        parser.add_argument('--bnn_prior_sigma_1', default=1.0, type=float, help='Prior sigma 1')
        parser.add_argument('--bnn_prior_sigma_2', default=0.002, type=float, help='Prior sigma 2')
        parser.add_argument('--bnn_prior_pi', default=0.5, type=float, help='Prior pi')
        parser.add_argument('--bnn_samples', default=10, type=int, help='Number of samples for MC dropout')
        return parser

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.dropout_rate = config.MODEL.DROP_RATE
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        
        # BNN parameters
        self.enable_bnn = config.MODEL.FactorizePhys.get("enable_bnn", False)
        self.bnn_kl_weight = config.MODEL.FactorizePhys.get("bnn_kl_weight", 1e-6)
        self.bnn_prior_sigma_1 = config.MODEL.FactorizePhys.get("bnn_prior_sigma_1", 1.0)
        self.bnn_prior_sigma_2 = config.MODEL.FactorizePhys.get("bnn_prior_sigma_2", 0.002)
        self.bnn_prior_pi = config.MODEL.FactorizePhys.get("bnn_prior_pi", 0.5)
        self.bnn_samples = config.MODEL.FactorizePhys.get("bnn_samples", 10)
        
        # Print BNN info if enabled
        if self.enable_bnn:
            print("Bayesian Neural Network Enabled")
            print(f"KL Weight: {self.bnn_kl_weight}")
            print(f"Prior Sigma 1: {self.bnn_prior_sigma_1}")
            print(f"Prior Sigma 2: {self.bnn_prior_sigma_2}")
            print(f"Prior Pi: {self.bnn_prior_pi}")
            print(f"MC Samples: {self.bnn_samples}")

        if torch.cuda.is_available() and config.NUM_OF_GPU_TRAIN > 0:
            dev_list = [int(d) for d in config.DEVICE.replace("cuda:", "").split(",")]
            self.device = torch.device(dev_list[0])     #currently toolbox only supports 1 GPU
            self.num_of_gpu = 1     #config.NUM_OF_GPU_TRAIN  # set number of used GPUs
        else:
            self.device = torch.device("cpu")  # if no GPUs set device is CPU
            self.num_of_gpu = 0  # no GPUs used

        frames = self.config.MODEL.FactorizePhys.FRAME_NUM
        in_channels = self.config.MODEL.FactorizePhys.CHANNELS
        model_type = self.config.MODEL.FactorizePhys.TYPE
        model_type = model_type.lower()

        md_config = {}
        md_config["FRAME_NUM"] = self.config.MODEL.FactorizePhys.FRAME_NUM
        md_config["MD_TYPE"] = self.config.MODEL.FactorizePhys.MD_TYPE
        md_config["MD_FSAM"] = self.config.MODEL.FactorizePhys.MD_FSAM
        md_config["MD_TRANSFORM"] = self.config.MODEL.FactorizePhys.MD_TRANSFORM
        md_config["MD_S"] = self.config.MODEL.FactorizePhys.MD_S
        md_config["MD_R"] = self.config.MODEL.FactorizePhys.MD_R
        md_config["MD_STEPS"] = self.config.MODEL.FactorizePhys.MD_STEPS
        md_config["MD_INFERENCE"] = self.config.MODEL.FactorizePhys.MD_INFERENCE
        md_config["MD_RESIDUAL"] = self.config.MODEL.FactorizePhys.MD_RESIDUAL
        
        # Add BNN parameters to config
        md_config["enable_bnn"] = self.enable_bnn
        md_config["bnn_prior_sigma_1"] = self.bnn_prior_sigma_1
        md_config["bnn_prior_sigma_2"] = self.bnn_prior_sigma_2
        md_config["bnn_prior_pi"] = self.bnn_prior_pi
        md_config["bnn_kl_weight"] = self.bnn_kl_weight

        self.md_infer = self.config.MODEL.FactorizePhys.MD_INFERENCE
        self.use_fsam = self.config.MODEL.FactorizePhys.MD_FSAM

        if model_type == "standard":
            self.model = FactorizePhys(frames=frames, md_config=md_config, in_channels=in_channels,
                                    dropout=self.dropout_rate, device=self.device)  # [3, T, 72,72]
        elif model_type == "big":
            self.model = FactorizePhysBig(frames=frames, md_config=md_config, in_channels=in_channels,
                                       dropout=self.dropout_rate, device=self.device)  # [3, T, 144,144]
        else:
            print("Unexpected model type specified. Should be standard or big, but specified:", model_type)
            exit()

        if torch.cuda.device_count() > 0 and self.num_of_gpu > 0:  # distribute model across GPUs
            self.model = torch.nn.DataParallel(self.model, device_ids=[self.device])  # data parallel model
        else:
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        if self.config.TOOLBOX_MODE == "train_and_test" or self.config.TOOLBOX_MODE == "only_train":
            self.num_train_batches = len(data_loader["train"])
            self.criterion = Neg_Pearson()
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.config.TRAIN.LR)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.config.TRAIN.LR, epochs=self.config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif self.config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("FactorizePhys trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        mean_training_losses = []
        mean_valid_losses = []
        mean_appx_error = []
        mean_kl_loss = []  # For tracking KL divergence loss
        lrs = []
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            appx_error_list = []
            kl_loss_list = []  # For tracking KL divergence loss
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                
                data = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                
                if len(labels.shape) > 2:
                    labels = labels[..., 0]     # Compatibility wigth multi-signal labelled data
                labels = (labels - torch.mean(labels)) / torch.std(labels)  # normalize
                last_frame = torch.unsqueeze(data[:, :, -1, :, :], 2).repeat(1, 1, max(self.num_of_gpu, 1), 1, 1)
                data = torch.cat((data, last_frame), 2)

                self.optimizer.zero_grad()
                
                # Forward pass - depends on model output format
                if self.model.training and self.use_fsam:
                    pred_ppg, factorized_embed, appx_error = self.model(data)
                else:
                    pred_ppg = self.model(data)
                
                pred_ppg = (pred_ppg - torch.mean(pred_ppg)) / torch.std(pred_ppg)  # normalize

                # Calculate standard loss
                neg_pearson_loss = self.criterion(pred_ppg, labels)
                
                # Calculate KL divergence loss if BNN is enabled
                kl_loss = torch.tensor(0.0).to(self.device)
                if self.enable_bnn:
                    # Get KL divergence from model
                    if hasattr(self.model.module, 'kl_divergence'):
                        kl_loss = self.model.module.kl_divergence()
                    else:
                        kl_loss = torch.tensor(0.0).to(self.device)
                    
                    # Track KL loss
                    kl_loss_list.append(kl_loss.item())
                    
                    # Total loss: negative Pearson + weighted KL divergence
                    # KL weight annealing: start from 0 and increase to target value
                    kl_weight = self.bnn_kl_weight * min(1.0, epoch / 10)
                    loss = neg_pearson_loss + kl_weight * kl_loss
                else:
                    # Standard loss (no BNN)
                    loss = neg_pearson_loss
                
                loss.backward()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                if self.use_fsam:
                    appx_error_list.append(appx_error.item())

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())

                self.optimizer.step()
                self.scheduler.step()
                
                # Set tbar postfix based on BNN mode
                if self.enable_bnn:
                    if self.use_fsam:
                        tbar.set_postfix({"neg_pearson": neg_pearson_loss.item(), "kl_loss": kl_loss.item(), "appx_error": appx_error.item()})
                    else:
                        tbar.set_postfix({"neg_pearson": neg_pearson_loss.item(), "kl_loss": kl_loss.item()})
                else:
                    if self.use_fsam:
                        tbar.set_postfix({"appx_error": appx_error.item()}, loss=loss.item())
                    else:
                        tbar.set_postfix(loss=loss.item())

            # Append the mean training loss for the epoch
            mean_training_losses.append(np.mean(train_loss))
            if self.use_fsam:
                mean_appx_error.append(np.mean(appx_error_list))
            if self.enable_bnn and kl_loss_list:
                mean_kl_loss.append(np.mean(kl_loss_list))
                print(f"Mean train loss: {np.mean(train_loss)}, Mean KL loss: {np.mean(kl_loss_list)}")
            elif self.use_fsam:
                print(f"Mean train loss: {np.mean(train_loss)}, Mean appx error: {np.mean(appx_error_list)}")
            else:
                print(f"Mean train loss: {np.mean(train_loss)}")

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")

                data, labels = valid_batch[0].to(self.device), valid_batch[1].to(self.device)
                if len(labels.shape) > 2:
                    labels = labels[..., 0]     # Compatibility wigth multi-signal labelled data
                labels = (labels - torch.mean(labels)) / torch.std(labels)  # normalize
                last_frame = torch.unsqueeze(data[:, :, -1, :, :], 2).repeat(1, 1, max(self.num_of_gpu, 1), 1, 1)
                data = torch.cat((data, last_frame), 2)

                # Run forward pass - use deterministic forward for validation when BNN is enabled
                if self.enable_bnn:
                    if hasattr(self.model.module, 'deterministic_forward'):
                        if self.use_fsam:
                            pred_ppg, _, _ = self.model.module.deterministic_forward(data)
                        else:
                            pred_ppg = self.model.module.deterministic_forward(data)
                    else:
                        # Fallback to regular forward if deterministic not available
                        if self.model.training and self.use_fsam:
                            pred_ppg, _, _ = self.model(data)
                        else:
                            pred_ppg = self.model(data)
                else:
                    # Standard forward pass
                    if self.model.training and self.use_fsam:
                        pred_ppg, _, _ = self.model(data)
                    else:
                        pred_ppg = self.model(data)
                
                pred_ppg = (pred_ppg - torch.mean(pred_ppg)) / torch.std(pred_ppg)  # normalize
                loss = self.criterion(pred_ppg, labels)
                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()
        uncertainty_predictions = dict()  # For storing multiple MC samples when BNN is enabled

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.device)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data, label = test_batch[0].to(self.device), test_batch[1].to(self.device)
                
                # Add an extra frame for compatibility with original code
                last_frame = torch.unsqueeze(data[:, :, -1, :, :], 2).repeat(1, 1, max(self.num_of_gpu, 1), 1, 1)
                data = torch.cat((data, last_frame), 2)
                
                # Standard deterministic prediction
                if self.enable_bnn:
                    if hasattr(self.model.module, 'deterministic_forward'):
                        if self.use_fsam:
                            det_pred_ppg, _, _ = self.model.module.deterministic_forward(data)
                        else:
                            det_pred_ppg = self.model.module.deterministic_forward(data)
                    else:
                        # Fallback
                        if self.use_fsam:
                            det_pred_ppg, _, _ = self.model(data)
                        else:
                            det_pred_ppg = self.model(data)
                else:
                    # Standard model
                    if self.use_fsam:
                        det_pred_ppg, _, _ = self.model(data)
                    else:
                        det_pred_ppg = self.model(data)
                
                # If BNN is enabled, run Monte Carlo sampling for uncertainty estimation
                if self.enable_bnn and hasattr(self.model.module, 'monte_carlo_forward'):
                    # Run multiple forward passes
                    mc_samples = self.model.module.monte_carlo_forward(data, n_samples=self.bnn_samples)
                
                # Prepare for storing in predictions dictionary
                if self.config.TEST.OUTPUT_SAVE_DIR:
                    label = label.cpu()
                    det_pred_ppg = det_pred_ppg.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                        if self.enable_bnn:
                            uncertainty_predictions[subj_index] = dict()
                    
                    predictions[subj_index][sort_index] = det_pred_ppg[idx]
                    labels[subj_index][sort_index] = label[idx]
                    
                    # Store MC samples for uncertainty visualization
                    if self.enable_bnn:
                        uncertainty_predictions[subj_index][sort_index] = [sample[idx].numpy() for sample in mc_samples]

        # Calculate standard metrics
        print('')
        calculate_metrics(predictions, labels, self.config)
        
        # Save outputs and generate uncertainty visualizations if requested
        if self.config.TEST.OUTPUT_SAVE_DIR:
            # Create visualization output directory
            vis_output_dir = os.path.join(self.config.TEST.OUTPUT_SAVE_DIR, 'visualizations')
            os.makedirs(vis_output_dir, exist_ok=True)
            
            # Generate uncertainty visualizations if BNN is enabled
            if self.enable_bnn:
                print("Generating uncertainty visualizations...")
                
                # For each subject, create uncertainty visualizations
                for subj_index in uncertainty_predictions.keys():
                    for sort_index in uncertainty_predictions[subj_index].keys():
                        # Get predictions and labels
                        mc_samples = uncertainty_predictions[subj_index][sort_index]
                        true_label = labels[subj_index][sort_index].numpy()
                        det_pred = predictions[subj_index][sort_index].numpy()
                        
                        # Generate filename ID
                        filename_id = f"{self.model_file_name}_{subj_index}_{sort_index}"
                        
                        # Plot confidence intervals
                        plot_confidence_intervals(
                            mc_samples, 
                            true_label, 
                            self.config.TEST.DATA.FS, 
                            vis_output_dir, 
                            filename_id
                        )
                        
                        # Plot BNN vs deterministic comparison
                        plot_bnn_vs_deterministic(
                            mc_samples,
                            det_pred,
                            true_label,
                            self.config.TEST.DATA.FS,
                            vis_output_dir,
                            filename_id
                        )
                        
                        # Plot HR estimation with uncertainty
                        plot_hr_estimation_with_uncertainty(
                            mc_samples,
                            true_label,
                            self.config.TEST.DATA.FS,
                            vis_output_dir,
                            filename_id
                        )
                        
                # Plot weight distributions for a few samples
                if hasattr(self.model.module, 'rppg_feature_extractor') and hasattr(self.model.module, 'rppg_head'):
                    # Find layer names for plotting
                    layer_names = []
                    for name, module in self.model.named_modules():
                        if hasattr(module, 'weight_mu') and hasattr(module, 'weight_rho'):
                            layer_names.append(name)
                    
                    # Select a subset of layers if there are many
                    if len(layer_names) > 6:
                        layer_names = layer_names[:3] + layer_names[-3:]
                    
                    # Plot weight distributions
                    if layer_names:
                        plot_weight_distributions(
                            self.model, 
                            layer_names, 
                            vis_output_dir,
                            self.model_file_name
                        )
            
            # Save standard test outputs
            self.save_test_outputs(predictions, labels, self.config)
            
            # If BNN is enabled, also save the uncertainty predictions
            if self.enable_bnn:
                self.save_uncertainty_outputs(uncertainty_predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
    
    def save_uncertainty_outputs(self, uncertainty_predictions, labels, config):
        """Save Monte Carlo samples for uncertainty visualization
        
        Args:
            uncertainty_predictions: Dictionary of Monte Carlo samples
            labels: Dictionary of ground truth labels
            config: Configuration object
        """
        import pickle
        
        output_dir = config.TEST.OUTPUT_SAVE_DIR
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Filename ID to be used in any output files that get saved
        if config.TOOLBOX_MODE == 'train_and_test':
            filename_id = self.model_file_name
        elif config.TOOLBOX_MODE == 'only_test':
            model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
            filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
        else:
            raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')
        
        output_path = os.path.join(output_dir, filename_id + '_uncertainty_outputs.pickle')

        data = dict()
        data['uncertainty_predictions'] = uncertainty_predictions
        data['labels'] = labels
        data['label_type'] = config.TEST.DATA.PREPROCESS.LABEL_TYPE
        data['fs'] = config.TEST.DATA.FS
        data['bnn_samples'] = self.bnn_samples

        with open(output_path, 'wb') as handle: # save out frame dict pickle file
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Saving uncertainty outputs to:', output_path)
