"""
Optuna optimization of the GNN.

# --- Last update 08/22/23 by Andrea Roncoli --- #

"""

# --- IMPORTS --- #
import datetime
import os
import random
import shutil
import time
from contextlib import redirect_stdout
from functools import partial

import numpy as np
import optuna
import psutil
import torch
from optuna.trial import TrialState
from optuna.visualization import (plot_contour, plot_optimization_history,
                                  plot_param_importances)

from scripts.constants import device, BATCH_SIZE
from scripts.early_stopping import EarlyStopping
from scripts.hyperparameters import HyperParameters
from scripts.load_data import create_dataset, split_datasets
from scripts.metalayer import define_model
from scripts.plotting import plot_losses, plot_out_true_scatter, plot_isomap
from scripts.training import evaluate, train, compute_encodings
from scripts.utils.logging_utils import general_log, initial_logging

# --- TRAINING CONSTANTS --- #

# OPTUNA CONSTANTS
N_TRIALS = 30                       # Number of trials
N_STARTUP_TRIALS = 20               # Number of startup trials with random sampling
N_NONPRUNED_TRIALS = 30             # Number of non-pruned trials 
PRUNING_PATIENCE = 20               # Patience for pruning (epochs)
EARLY_STOPPING_PATIENCE = 100       # Patience for early stopping (epochs)
EARLY_STOPPING_TOLERANCE = 1.e-4    # Tolerance for early stopping


# TRIAL CONSTANTS - Non-optimizable hyperparameters (default values for construction choices)
SIMSUITE = "IllustrisTNG"              # Simulation suite, choose between "IllustrisTNG" and "SIMBA"
SIMSET = "LH"                       # Simulation set, choose between "CV" and "LH"
N_SIMS = 1000                       # Number of simulations considered, maximum 27 for CV and 1000 for LH
DOMAIN_ADAPT = 'MMD'                # Domain Adaptation type
TRAINING = True                     # If training, set to True, otherwise loads a pretrained model and tests it
PRED_PARAMS = 1                     # Number of cosmo/astro params to be predicted, starting from Omega_m, sigma_8, etc.
ONLY_POSITIONS = 0                  # 1 for using only positions as features, 0 for using additional galactic features
SNAP = "33"                         # Snapshot of the simulation, indicating redshift 4: z=3, 10: z=2, 14: z=1.5, 18: z=1, 24: z=0.5, 33: z=0
DA_LOSS_FRACTION = 0.4              # Fraction of the loss to be domain adaptation loss

# TRIAL DEFAULTS - Optimizable hyperparameters
R_LINK = 0.015                      # Linking radius to build the graph
N_LAYERS = 2                        # Number of graph layers
HIDDEN_CHANNELS = 64                # Hidden channels
N_EPOCHS = 500                      # Number of epochs
LEARNING_RATE = 1e-7                # Learning rate
WEIGHT_DECAY = 1e-7                 # Weight decay
WEIGHT_DA = 1e-1                    # Domain adaptation weight
MAX_LR = 1e-3                       # Maximum learning rate for the cyclic learning rate scheduler
NUM_CYCLES = 2                      # Number of cycles for the cyclic learning rate scheduler
CYCLE_TYPE = "triangular"           # Type of cycle for the cyclic learning rate scheduler, either "triangular" or "triangular2"

params_values = [SIMSUITE, SIMSET, N_SIMS, DOMAIN_ADAPT, TRAINING, PRED_PARAMS, ONLY_POSITIONS, SNAP, DA_LOSS_FRACTION,\
                R_LINK, N_LAYERS, HIDDEN_CHANNELS, N_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, WEIGHT_DA]
params_keys = ["simsuite", "simset", "n_sims", "domain_adapt", "training", "pred_params", "only_positions", "snap", "da_loss_fraction",\
                "r_link", "n_layers", "hidden_channels", "n_epochs", "learning_rate", "weight_decay", "weight_da"]


# --- PARAMETERS TO OPTIMIZE --- #
# Comment out the ones you don't want to optimize, which will be set to the default value above
optimize_params = [     #"r_link",
                        #"n_layers",
                        #"hidden_channels",
                        #"n_epochs",
                        "learning_rate",
                        "weight_decay",
                        "weight_da"
                    ]

# --- HYPERPARAMETERS BOUNDS --- #
# Bounds for the hyperparameters suggested by optuna
min_rlink = 5.e-3
max_rlink = 3.e-2
min_nlayers = 1
max_nlayers = 5
hidden_channels_options = [32, 64, 128]
min_nepochs = 300
max_nepochs = 600
min_learning_rate = 1.e-8
max_learning_rate = 1.e-4
min_weight_decay = 1.e-8
max_weight_decay = 1.e-3
min_weight_da = 1.e-3
max_weight_da = 1.e3


# --- FUNCTIONS --- #
""" 
define_model(trial) should return a model with the hyperparameters defined by the trial
set_hparams(trial, hparams) should set the hyperparameters of hparams to the values defined by the trial
objective(trial) should return the accuracy of the model defined by the trial
"""

def set_hparams(trial, optimize_params):
    """Set the hyperparameters of hparams to the values defined by the trial.
    Checks which are to optimize by looking at optimize_params.

    Args:
        trial (optuna.Trial): Trial object.
        optimize_params (list): List of parameters to optimize that optuna will suggest.
        suggest (bool): If True, optuna will suggest the parameters, otherwise they will be set to the default values.

    Returns:
        HyperParameters: Hyperparameters.
    """
    hparams_dict = dict(zip(params_keys, params_values))

    if "r_link" in optimize_params:
        hparams_dict["r_link"] = trial.suggest_float("r_link", min_rlink, max_rlink, log=True)
    if "n_layers" in optimize_params:
        hparams_dict["n_layers"] = trial.suggest_int("n_layers", min_nlayers, max_nlayers)
    if "hidden_channels" in optimize_params:
        hparams_dict["hidden_channels"] = trial.suggest_categorical("hidden_channels", hidden_channels_options)
    if "n_epochs" in optimize_params:
        hparams_dict["n_epochs"] = trial.suggest_int("n_epochs", min_nepochs, max_nepochs)
    if "learning_rate" in optimize_params:
        hparams_dict["learning_rate"] = trial.suggest_float("learning_rate", min_learning_rate, max_learning_rate, log=True)
    if "weight_decay" in optimize_params:
        hparams_dict["weight_decay"] = trial.suggest_float("weight_decay", min_weight_decay, max_weight_decay, log=True)
    if "weight_da" in optimize_params:
        hparams_dict["weight_da"] = trial.suggest_float("weight_da", min_weight_da, max_weight_da, log=True)

    hparams = HyperParameters(**hparams_dict)

    return hparams

def objective_function(trial, optimize_params, verbose = True):
    # Print for keeping Jupyter server alive
    print('Keeping server alive...: Starting trial number {}!'.format(trial.number))

    # Set the hyperparameters
    hparams = set_hparams(trial, optimize_params)

    # Write a file that serves as checkpoint to know where we are
    with open("checkpoints/checkpoint_trial_"+str(trial.number)+".txt", "w") as checkpoint:
        checkpoint.close()

    # Set plotting directory with trial number for easy identification
    plot_dir = "Plots/trial_"+str(trial.number)+"/"
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    with open("logs/log_trial_{}_{}.txt".format(trial.number, hparams.name_model()), "w") as logfile:
        # Everything printed in the console will be written in the file
        with redirect_stdout(logfile):
            time_init_trial = time.time()
            
            # Some logging info
            print("Trial number: {}".format(trial.number))
            print("Trial name: {}".format(hparams.name_model()))
            print("Trial params: {}".format(hparams))
            print("Optimizing params: {}".format(optimize_params))

            # Create the dataset
            if verbose: 
                print("Creating loaders... ")
                time_ini = time.time()

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
            
            # Define optimizer and learning rate scheduler
            optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)

            n_iterations = N_SIMS / BATCH_SIZE * hparams.n_epochs

            # Option 1: Cyclic learning rate scheduler 
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=hparams.learning_rate, max_lr=MAX_LR,\
                                                            step_size_up=n_iterations/(NUM_CYCLES*2), step_size_down=n_iterations/(NUM_CYCLES*2),\
                                                            cycle_momentum=False, mode=CYCLE_TYPE)

            # Option 2: Cosine annealing learning rate scheduler
            #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iterations, eta_min=1.e-8)

            # Initialize early stopping object
            early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode='min', tolerance=EARLY_STOPPING_TOLERANCE)

            train_losses, valid_losses = [], []
            train_losses_mmd, valid_losses_mmd = [], []
            valid_loss_min = 1000.

            # Training routine
            for epoch in range(1, hparams.n_epochs+1):
                epoch_start_time = time.time()
                train_loss, train_loss_mmd = train(train_loader, model, hparams, optimizer, scheduler)
                valid_loss, valid_loss_mmd, mean_abs_error = evaluate(valid_loader, model, hparams)

                trial.report(valid_loss, epoch)

                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    print("Trial pruned at epoch {}".format(epoch))
                    raise optuna.exceptions.TrialPruned()

                train_losses.append(train_loss) 
                valid_losses.append(valid_loss) 
                train_losses_mmd.append(train_loss_mmd)
                valid_losses_mmd.append(valid_loss_mmd)

                if early_stopping(valid_loss):
                    print("Early stopping at epoch {}".format(epoch))
                    break

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
                    plot_isomap(source_encodings, target_encodings, labels, epoch, hparams, n_components = 2, dir = plot_dir)

            plot_losses(train_losses, valid_losses, hparams, dir = plot_dir)
            plot_losses(train_losses_mmd, valid_losses_mmd, hparams, plot_mmd=True, dir = plot_dir)

            # Load the trained model
            state_dict = torch.load("Models/"+hparams.name_model(), map_location=device)
            model.load_state_dict(state_dict)

            # Evaluation on validation set of model saved at best validation error
            for same_suite in [True, False]:
                valid_loss, mmd_loss, _ = evaluate(valid_loader, model, hparams, same_suite)

                if same_suite: 
                    print("Validation loss on same suite: {:.2e}".format(valid_loss))
                    print("Validation MMD-only loss on same suite: {:.2e}".format(mmd_loss))
                    valid_loss_same = valid_loss
                else:
                    print("Validation loss on opposite suite: {:.2e}".format(valid_loss))
                    print("Validation MMD-only loss on opposite suite: {:.2e}".format(mmd_loss))

                # Plot true vs predicted cosmo parameters
                plot_out_true_scatter(hparams, "Om", same_suite = same_suite, dir = plot_dir)
                if hparams.pred_params==2:
                    plot_out_true_scatter(hparams, "Sig", same_suite = same_suite, dir = plot_dir)

            print("Time elapsed for trial: {:.2f} s".format(time.time()-time_init_trial))

        logfile.close()

    return valid_loss_same


# --- MAIN --- #
    """Optimize the hyperparameters of the GNN.
    """

if __name__ == "__main__":

    time_ini_study = time.time()

    # Create the folders for storing the plots, models and outputs
    for path in ["Plots", "Models", "Outputs", "Plots/study"]:
        if not os.path.exists(path):
            os.mkdir(path)

    # If there isn't a logs folder, create it
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Create a new checkpoint folder, deleting the previous one if it exists
    if os.path.exists('checkpoints'):
        shutil.rmtree('checkpoints')
    os.makedirs('checkpoints')

    # Optuna variables to create the study
    storage = "sqlite:///"+os.getcwd()+"/optuna_"+SIMSUITE
    study_name = "GNN_"+SIMSUITE


    objective = partial(objective_function, optimize_params = optimize_params)
    sampler = optuna.samplers.TPESampler(n_startup_trials = N_STARTUP_TRIALS)

    pruner = optuna.pruners.MedianPruner(n_startup_trials = N_NONPRUNED_TRIALS)
    patient_pruner = optuna.pruners.PatientPruner(wrapped_pruner = pruner, patience = PRUNING_PATIENCE)
    
    study = optuna.create_study(direction = 'minimize', study_name=study_name, sampler=sampler, storage=storage, pruner = patient_pruner, load_if_exists=True)

    initial_logging(params_keys, params_values, optimize_params, device, study)
    general_log("DA fraction:", DA_LOSS_FRACTION)
    general_log("Number of cycles:", NUM_CYCLES)
    general_log("Maximum learning rate:", MAX_LR)
    general_log("Cycle mode:", CYCLE_TYPE)

    study.optimize(objective, n_trials = N_TRIALS, gc_after_trial=True, show_progress_bar=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Print info for the study
    general_log("Study statistics: ")
    general_log("  Number of finished trials: {}".format(len(study.trials)))
    general_log("  Number of pruned trials: {}".format(len(pruned_trials)))
    general_log("  Number of complete trials: {}".format(len(complete_trials)))

    # Print info for best trial
    trial = study.best_trial

    general_log("Best trial:")
    general_log("  Value: {}".format(trial.value))
    general_log("  Params: ")

    for key, value in trial.params.items():
        general_log("    {}: {}".format(key, value))

    # Save best model and plots
    if not os.path.exists("Best"):
        os.mkdir("Best")

    # Reconstruct an hparams object to get the name for the best trial
    hparams = HyperParameters(**dict(zip(params_keys, params_values)))

    # Set the hyperparameters attributes, remember that trial.params and hparams have the same keys
    for key, value in trial.params.items():
        setattr(hparams, key, value)
    
    plot_dir = "Plots/trial_"+str(trial.number)+"/"

    files = []
    files.append( plot_dir+hparams.name_model()+"_out_true_Om_validset.png" )
    files.append( plot_dir+hparams.name_model()+"_out_true_Om_testsuite_validset.png")
    files.append( plot_dir+hparams.name_model()+"_out_true_Sig_validset.png" )
    files.append( plot_dir+hparams.name_model()+"_out_true_Sig_testsuite_validset.png")
    files.append( plot_dir+hparams.name_model()+"_loss.png" )
    files.append( plot_dir+hparams.name_model()+"_loss_mmd.png" )
    files.append( "Models/"+hparams.name_model() )
    for epoch in range(1, hparams.n_epochs+1):
        files.append( plot_dir+hparams.name_model()+"_isomap_2D_epochs_"+str(epoch)+".png" )

    for file in files:
        if os.path.exists(file):
            os.system("cp "+file+" Best/.")

    # Visualization of optimization results
    fig = plot_optimization_history(study)
    fig.write_image("Plots/study/optuna_optimization_history_"+SIMSUITE+".png")

    fig = plot_contour(study)#, params=["learning_rate", "weight_decay", "r_link"])#, "use_model"])
    fig.write_image("Plots/study/optuna_contour_"+SIMSUITE+".png")

    fig = plot_param_importances(study)
    fig.write_image("Plots/study/plot_param_importances_"+SIMSUITE+".png")

    general_log("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini_study))
