""" Routine to make the assessment of the model.
Initially a set of hparams is chosen and the model is trained.
The model with lowest validation during the run will be saved.
The model is then assessed on the test set.

The values for assessment are hardcoded above the main function.
"""

from scripts.constants import *
from scripts.load_data import create_dataset, split_datasets
from scripts.metalayer import define_model
from scripts.training import train, evaluate, compute_encodings
from scripts.plotting import plot_losses, plot_out_true_scatter, plot_isomap
from scripts.hyperparameters import HyperParameters

import time
import os
import psutil
from contextlib import redirect_stdout

ISOMAP_ON_ALL_DATA = True           # If True, the isomap will be computed on the whole dataset, not only the test set
PRETRAINED_MODEL = False

# TRIAL CONSTANTS - Non-optimizable hyperparameters (default values for construction choices)
SIMSUITE = "IllustrisTNG"           # Simulation suite, choose between "IllustrisTNG" and "SIMBA"
SIMSET = "LH"                       # Simulation set, choose between "CV" and "LH"
N_SIMS = 1000                       # Number of simulations considered, maximum 27 for CV and 1000 for LH
DOMAIN_ADAPT = 'MMD'                # Domain Adaptation type
TRAINING = True                     # If training, set to True, otherwise loads a pretrained model and tests it
PRED_PARAMS = 1                     # Number of cosmo/astro params to be predicted, starting from Omega_m, sigma_8, etc.
ONLY_POSITIONS = 0                  # 1 for using only positions as features, 0 for using additional galactic features
SNAP = "33"                         # Snapshot of the simulation, indicating redshift 4: z=3, 10: z=2, 14: z=1.5, 18: z=1, 24: z=0.5, 33: z=0
DA_LOSS_FRACTION = 0.4              # Fraction of the loss to be domain adaptation loss

# TRIAL DEFAULTS - Optimizable hyperparameters
N_EPOCHS = 500                      # Number of epochs
MAX_LR = 1e-3                       # Maximum learning rate for the cyclic learning rate scheduler
NUM_CYCLES = 2.75                   # Number of cycles for the cyclic learning rate scheduler
CYCLE_TYPE = "triangular"           # Type of cycle for the cyclic learning rate scheduler, either "triangular" or "triangular2"

#################### CHOOSE WHAT RESULTS TO REPLICATE ####################
# Uncomment the parameters you want to use, and comment out the others

#"""

# A) Illustris with MMD

SIMSUITE = "IllustrisTNG"
DOMAIN_ADAPT = "MMD"
R_LINK = 0.015
N_LAYERS = 2
HIDDEN_CHANNELS = 64

# Hyperparameters found from optimization run:
# Learning rate: 2.0663951428812126e-06
# Weight decay: 3.0019250983369906e-06
# Domain adaptation weight: 0.1080717661141762

LEARNING_RATE = 2.0663951428812126e-06
WEIGHT_DECAY = 3.0019250983369906e-06
WEIGHT_DA = 0.1080717661141762
"""
# ---------------------------- #
# B) SIMBA with MMD

SIMSUITE = "SIMBA"
DOMAIN_ADAPT = "MMD"
R_LINK = 0.0148
N_LAYERS = 4
HIDDEN_CHANNELS = 64

# Hyperparameters found from optimization run: 
# Learning rate: 6.694651038384752e-08
# Weight decay: 1.4972695006384463e-05
# Domain adaptation weight: 0.11086960006145966

LEARNING_RATE = 6.694651038384752e-08
WEIGHT_DECAY = 1.4972695006384463e-05
WEIGHT_DA = 0.11086960006145966

# ---------------------------- #
# C) Illustris without domain adaptation

SIMSUITE = "IllustrisTNG"
DOMAIN_ADAPT = "None"
R_LINK = 0.015
N_LAYERS = 2
HIDDEN_CHANNELS = 64

# Hyperparameters from CosmoGraphNet paper:
# Learning rate: 1.619e-07
# Weight decay: 1e-07

LEARNING_RATE = 1.619e-07
WEIGHT_DECAY = 1e-07

# ---------------------------- #
# D) SIMBA without domain adaptation

SIMSUITE = "SIMBA"
DOMAIN_ADAPT = "None"
R_LINK = 0.0148
N_LAYERS = 4
HIDDEN_CHANNELS = 64

# Hyperparameters from CosmoGraphNet paper:
# Learning rate: 1.087e-06
# Weight decay: 1e-07

LEARNING_RATE = 1.087e-06
WEIGHT_DECAY = 1e-07

"""

params_values = [SIMSUITE, SIMSET, N_SIMS, DOMAIN_ADAPT, TRAINING, PRED_PARAMS, ONLY_POSITIONS, SNAP, DA_LOSS_FRACTION,\
                R_LINK, N_LAYERS, HIDDEN_CHANNELS, N_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, WEIGHT_DA]
params_keys = ["simsuite", "simset", "n_sims", "domain_adapt", "training", "pred_params", "only_positions", "snap", "da_loss_fraction",\
                "r_link", "n_layers", "hidden_channels", "n_epochs", "learning_rate", "weight_decay", "weight_da"]


def main(hparams, verbose = True):
    # Set plotting directory with trial number for easy identification
    plot_dir = "Plots/"

    with open("logs/log_assessment_{}.txt".format(hparams.name_model()), "w") as logfile:
        # Everything printed in the console will be written in the file
        with redirect_stdout(logfile):        
            
            if verbose: 
                print("Assessment of model {}...".format(hparams.name_model()))
                time_ini = time.time()
            
            # Create the dataset
            if verbose: 
                print("Creating loaders... ")

            # Create the dataset
            datasets = create_dataset(hparams)

            # Split dataset among training, validation and testing datasets
            train_loader = {}
            valid_loader = {} 
            test_loader = {}
            for name, dataset in datasets.items():
                train_loader[name], valid_loader[name], test_loader[name] = split_datasets(dataset)

            if verbose: 
                time_end = time.time()
                print("Time to create loaders: {:.2f} s".format(time_end-time_ini))

            # Define the model
            if datasets[hparams.simsuite][0].x.shape[1] != datasets[hparams.flip_suite()][0].x.shape[1]:
                raise ValueError("The number of features for the two must be the same, but are {} for {} and {} for {}"\
                                 .format(train_loader[hparams.simsuite][0].x.shape[1], hparams.simsuite,\
                                         train_loader[hparams.flip_suite()][0].x.shape[1], hparams.flip_suite()))
            
            dim_in = datasets[hparams.simsuite][0].x.shape[1]
            dim_out = hparams.pred_params * 2
            model = define_model(hparams, dim_in, dim_out)
            model.to(device)

            if verbose:
                # Print the memory (in GB) being used now:
                process = psutil.Process()
                print("Memory being used (GB):",process.memory_info().rss/1.e9)
            
            if not PRETRAINED_MODEL:
                # Define optimizer and learning rate scheduler
                optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)

                n_iterations = N_SIMS / BATCH_SIZE * hparams.n_epochs

                # Option 1: Cyclic learning rate scheduler 
                scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=hparams.learning_rate, max_lr=MAX_LR,\
                                                                step_size_up=n_iterations/(NUM_CYCLES*2), step_size_down=n_iterations/(NUM_CYCLES*2),\
                                                                cycle_momentum=False, mode=CYCLE_TYPE)

                train_losses, valid_losses = [], []
                train_losses_mmd, valid_losses_mmd = [], []
                valid_loss_min, err_min = 1000., 1000.

                # Training routine
                for epoch in range(1, hparams.n_epochs+1):
                    epoch_start_time = time.time()
                    train_loss, train_loss_mmd = train(train_loader, model, hparams, optimizer, scheduler)
                    valid_loss, valid_loss_mmd, mean_abs_error = evaluate(valid_loader, model, hparams)

                    train_losses.append(train_loss) 
                    valid_losses.append(valid_loss) 
                    train_losses_mmd.append(train_loss_mmd)
                    valid_losses_mmd.append(valid_loss_mmd)

                    # Save model if it has improved
                    if valid_loss <= valid_loss_min:
                        if verbose: print("Validation loss decreased ({:.2e} --> {:.2e}), on epoch {}. Saving model ...".format(valid_loss_min, valid_loss, epoch))
                        torch.save(model.state_dict(), "Models/"+hparams.name_model())

                        valid_loss_min = valid_loss

                    # Print training/validation statistics
                    epoch_end_time = time.time()

                    if verbose: print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.2e}, Validation Loss: {valid_loss:.2e}, Mean Absolute Error: {mean_abs_error:.2e}, Time: {epoch_end_time-epoch_start_time:.2f}s')

                    # Plot the isomap for domain adaptation interpretation
                    if epoch % 50 == 0 or epoch == 1:
                        source_encodings, target_encodings, labels = compute_encodings(train_loader, model)
                        plot_isomap(source_encodings, target_encodings, labels, epoch, hparams, n_components = 2)

                plot_losses(train_losses, valid_losses, hparams)
                plot_losses(train_losses_mmd, valid_losses_mmd, hparams, plot_mmd=True)

                # Load the trained model
                state_dict = torch.load("Models/"+hparams.name_model(), map_location=device)
            else: 
                state_dict  = torch.load("ModelsPreTrained/"+hparams.name_model(), map_location=device)

            model.load_state_dict(state_dict)            

            # Evaluation on test set of model saved at best validation error
            for same_suite in [True, False]:
                test_loss, mmd_loss, _ = evaluate(test_loader, model, hparams, same_suite)

                if same_suite: 
                    print("Test loss on same suite: {:.2e}".format(test_loss))
                    print("Test MMD-only loss on same suite: {:.2e}".format(mmd_loss))
                else:
                    print("Test loss on opposite suite: {:.2e}".format(test_loss))
                    print("Test MMD-only loss on opposite suite: {:.2e}".format(mmd_loss))

                # Plot true vs predicted cosmo parameters
                plot_out_true_scatter(hparams, "Om", same_suite = same_suite, test = True)
                if hparams.pred_params==2:
                    plot_out_true_scatter(hparams, "Sig", same_suite = same_suite, test = True)

            # Plot the isomap for domain adaptation interpretation
            if ISOMAP_ON_ALL_DATA:
                source_encodings = np.empty((0, model.encoding_dim))
                target_encodings = np.empty((0, model.encoding_dim))
                labels = np.empty((2, 0, 1))
                for loader in [train_loader, valid_loader, test_loader]:
                    source_encodings_, target_encodings_, labels_ = compute_encodings(loader, model)

                    source_encodings = np.concatenate((source_encodings, source_encodings_), dtype=np.float32)
                    target_encodings = np.concatenate((target_encodings, target_encodings_), dtype=np.float32)
                    labels = np.concatenate((labels, labels_), axis=1, dtype=np.float32)
            else:
                source_encodings, target_encodings, labels = compute_encodings(test_loader, model)

            plot_isomap(source_encodings, target_encodings, labels, N_EPOCHS, hparams, n_components = 2, dir = plot_dir, assessment=True)

            if verbose: 
                time_end = time.time()
                print("Total time: {:.2f} s".format(time_end-time_ini))

        logfile.close()


if __name__ == "__main__":

    # Create the folders for storing the plots, models and outputs
    for path in ["Plots", "Models", "Outputs"]:
        if not os.path.exists(path):
            os.mkdir(path)

    # If there isn't a logs folder, create it
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Construct hyperparameters from the lists above
    hparams_dict = dict(zip(params_keys, params_values))
    hparams = HyperParameters(**hparams_dict)
    
    main(hparams)
