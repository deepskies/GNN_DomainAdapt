""" 
Plotting functions for the cosmological parameter estimation project.

"""

import time

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from .constants import device
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits import mplot3d
from sklearn.manifold import Isomap
from sklearn.metrics import r2_score

from .losses_da import mmd_distance

mpl.rcParams.update({'font.size': 12})

# Choose color depending on the CAMELS simulation suite
def colorsuite(suite):
    if suite=="IllustrisTNG":   
        return "purple"
    elif suite=="SIMBA":            
        return "dodgerblue"

# Plot loss trends
def plot_losses(train_losses, valid_losses, hparams, plot_mmd = False, test_loss = None, mean_abs_error = None, dir = "Plots/"):

    epochs = len(train_losses)
    #plt.plot(range(epochs), np.exp(train_losses), "r-",label="Training")
    #plt.plot(range(epochs), np.exp(valid_losses), "b:",label="Validation")
    plt.plot(range(epochs), train_losses, "r-",label="Training")
    plt.plot(range(epochs), valid_losses, "b:",label="Validation")

    plt.legend()
    #plt.yscale("log")
    if test_loss == None and mean_abs_error == None:
        if plot_mmd:
            plt.title("Training and validation MMD loss")
        else:
            plt.title("Training and validation loss")
    else:
        plt.title(f"Test loss: {test_loss:.2e}, Mean Absolute Error: {mean_abs_error:.2e}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()

    if plot_mmd:
        plt.savefig(dir+hparams.name_model()+"_loss_mmd.png", bbox_inches='tight', dpi=300)
    else:
        plt.savefig(dir+hparams.name_model()+"_loss.png", bbox_inches='tight', dpi=300)
    plt.close()

# Remove normalization of cosmo parameters
def denormalize_outputs(trues, outputs, errors, minpar, maxpar):

    trues = minpar + trues*(maxpar - minpar)
    outputs = minpar + outputs*(maxpar - minpar)
    errors = errors*(maxpar - minpar)
    return trues, outputs, errors

# Scatter plot of true vs predicted cosmological parameter
def plot_out_true_scatter(hparams, cosmoparam, same_suite = True, test= False, dir = "Plots/"):

    figscat, axscat = plt.subplots(figsize=(6,5))
    suite, simset = hparams.simsuite, hparams.simset

    # Load true values and predicted means and standard deviations
    if same_suite:
        print("Loading outputs from same suite as training")
        outputs = np.load("Outputs/"+hparams.name_model()+"_outputs_samesuite.npy")
        trues = np.load("Outputs/"+hparams.name_model()+"_trues_samesuite.npy")
        errors = np.load("Outputs/"+hparams.name_model()+"_errors_samesuite.npy")
        col = colorsuite(suite)
    else:
        print("Loading outputs from different suite as training")
        outputs = np.load("Outputs/"+hparams.name_model()+"_outputs_testsuite.npy")
        trues = np.load("Outputs/"+hparams.name_model()+"_trues_testsuite.npy")
        errors = np.load("Outputs/"+hparams.name_model()+"_errors_testsuite.npy")
        col = colorsuite(hparams.flip_suite())

    # There is a (0,0) initial point, fix it
    outputs = outputs[1:]
    trues = trues[1:]
    errors = errors[1:]

    # --------- Talked with Francisco Villanueva -------- #
    # Force std deviations to be positive to plot
    # Model is agnostic to sign of std, so this is introducing no particular loss of generality
    errors = np.abs(errors)
    # --------- Talked with Francisco Villanueva -------- #

    # Choose cosmo param and denormalize
    if cosmoparam=="Om":
        minpar, maxpar = 0.1, 0.5
        outputs, trues, errors = outputs[:,0], trues[:,0], errors[:,0]
    elif cosmoparam=="Sig":
        minpar, maxpar = 0.6, 1.0
        outputs, trues, errors = outputs[:,1], trues[:,1], errors[:,1]
    trues, outputs, errors = denormalize_outputs(trues, outputs, errors, minpar, maxpar)
    
    # Compute the number of points lying within 1 or 2 sigma regions from their uncertainties
    cond_success_1sig, cond_success_2sig = np.abs(outputs-trues)<=np.abs(errors), np.abs(outputs-trues)<=2.*np.abs(errors)
    tot_points = outputs.shape[0]
    successes1sig, successes2sig = outputs[cond_success_1sig].shape[0], outputs[cond_success_2sig].shape[0]

    # Compute the linear correlation coefficient
    r2 = r2_score(trues,outputs)
    err_rel = np.mean(np.abs((trues - outputs)/(trues)), axis=0)
    chi2s = (outputs-trues)**2./errors**2.
    chi2 = chi2s[chi2s<1.e4].mean()    # Remove some outliers which make explode the chi2
    print("R^2={:.2f}, Relative error={:.2e}, Chi2={:.2f}".format(r2, err_rel, chi2))
    print("A fraction of successes of", successes1sig/tot_points, "at 1 sigma,", successes2sig/tot_points, "at 2 sigmas")

    # Sort by true value
    indsort = trues.argsort()
    outputs, trues, errors = outputs[indsort], trues[indsort], errors[indsort]

    # Compute mean and std region within several bins
    truebins, binsize = np.linspace(trues[0], trues[-1], num=10, retstep=True)
    means, stds = [], []
    for i, bin in enumerate(truebins[:-1]):
        cond = (trues>=bin) & (trues<bin+binsize)
        outbin = outputs[cond]
        if len(outbin)==0:
            outmean, outstd = np.nan, np.nan    # Avoid error message from some bins without points
        else:
            outmean, outstd = outbin.mean(), outbin.std()
        means.append(outmean); stds.append(outstd)
    means, stds = np.array(means), np.array(stds)
    print("Std in bins:",stds[~np.isnan(stds)].mean(),"Mean predicted uncertainty:", np.abs(errors.mean()))

    # Plot predictions vs true values
    #truemin, truemax = trues.min(), trues.max()
    truemin, truemax = minpar-0.05, maxpar+0.05
    #axscat.plot([truemin, truemax], [0., 0.], "r-")
    #axscat.errorbar(trues, outputs-trues, yerr=errors, color=col, marker="o", ls="none", markersize=0.5, elinewidth=0.5, zorder=10)
    axscat.plot([truemin, truemax], [truemin, truemax], "r-")
    axscat.errorbar(trues, outputs, yerr=errors, color=col, marker="o", ls="none", markersize=0.5, elinewidth=0.5, zorder=10)

    # Legend
    if cosmoparam=="Om":
        par = "\t"+r"$\Omega_m$"
    elif cosmoparam=="Sig":
        par = "\t"+r"$\sigma_8$"
    leg = par+"\n"+"$R^2$={:.2f}".format(r2)+"\n"+"$\epsilon$={:.1f} %".format(100.*err_rel)+"\n"+"$\chi^2$={:.2f}".format(chi2)
    #leg = par+"\n"+"$R^2$={:.2f}".format(r2)+"\n"+"$\epsilon$={:.2e}".format(err_rel)
    at = AnchoredText(leg, frameon=True, loc="upper left")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axscat.add_artist(at)

    # Labels etc
    axscat.set_xlim([truemin, truemax])
    axscat.set_ylim([truemin, truemax])
    if not same_suite:
        axscat.set_ylim([truemin, truemax+0.1])
    axscat.set_ylabel(r"Prediction")
    axscat.set_xlabel(r"Truth")
    axscat.grid()

    # Title, indicating which are the training and testing suites
    if hparams.only_positions:
        usefeatures = "Only positions"
    else:
        usefeatures = r"Positions + $V_{\rm max}, M_*, R_*, Z_*$"
    props = dict(boxstyle='round', facecolor='white')#, alpha=0.5)
    axscat.text((minpar + maxpar)/2-0.02, minpar, usefeatures, color="k", bbox=props )

    print("Check on plot_out_true_scatter if model is IllustrisTNG: ", hparams.simsuite)
    if not same_suite:
        titlefig = "Training in "+suite+", testing in "+hparams.flip_suite()
        #titlefig = "Cross test in "+suite+usefeatures
        namefig = hparams.name_model() + "_out_true_" + cosmoparam+"_testsuite"
    else:
        titlefig = "Training in "+suite+", testing in "+suite
        #titlefig = "Train in "+suite+usefeatures
        namefig = hparams.name_model() + "_out_true_" + cosmoparam
    axscat.set_title(titlefig)

    if test:    
        namefig += "_testset"
    else:
        namefig += "_validset"

    figscat.savefig(dir+namefig+".png", bbox_inches='tight', dpi=300)
    plt.close(figscat)


def denormalize_labels(labels, minpar, maxpar):
    """Denormalize labels.

    Args:
        labels (np.array): labels to denormalize
        minpar (float): minimum value of the parameter
        maxpar (float): maximum value of the parameter

    Returns:
        np.array: denormalized labels
    """
    return minpar + labels*(maxpar - minpar)
    
def plot_isomap(source, target, labels, epochs, hparams, n_components = 2, dir = "Plots/", assessment = False):
    """Generate and save isomap plot of the source and target domains.

    Args:
        source (np.array): source domain
        target (np.array): target domain
        labels (list): list of labels for the source and target domains
        epochs (int): number of epochs
        hparams (HyperParameters): hyperparameters
        n_components (int): number of components for isomap
    """
    # Keep track of time
    if assessment:
        print("Plotting isomap for assessment...")
    else:
        print("Plotting isomap at epoch {}...".format(epochs))

    # Quantify distance among source and target distributions with mmd and js divergence
    mmd = np.sqrt(mmd_distance(torch.from_numpy(source).to(device), torch.from_numpy(target).to(device)).item())

    # If there are NaN values in source or target arrays, don't plot
    if np.any(np.isnan(source)) or np.any(np.isnan(target)):
        print("There are NaN values in source or target arrays. Skipping isomap plot...")
        return
    
    # Concatenate source and target distributions to generate isomap
    distributions = np.concatenate((source, target), axis = 0)
    
    # Generate isomap
    mapper = Isomap(n_components = n_components, n_neighbors = 20)
    mapped = mapper.fit_transform(distributions)

    # Split mapped distributions into source and target
    source_mapped = mapped[:source.shape[0]]
    target_mapped = mapped[source.shape[0]:]

    # Denormalize labels (WARNING: written for Om case)
    min_label, max_label = 0.1, 0.5
    source_labels = denormalize_labels(labels[0], min_label, max_label)
    target_labels = denormalize_labels(labels[1], min_label, max_label)

    # Remove outliers to aid good visualization and remove corresponding labels
    source_mapped, source_labels = remove_outliers(source_mapped, source_labels)
    target_mapped, target_labels = remove_outliers(target_mapped, target_labels)
    
    removed = distributions.shape[0] - source_mapped.shape[0] - target_mapped.shape[0]
    print("Number of outliers removed from isomap for visualization: ", removed)

    if n_components == 2:  

        fig = plt.figure(figsize=(6,5))
        if assessment:
            plt.title('Isomap 2D assessment')
        else:
            plt.title('Isomap 2D at epoch {}'.format(epochs))

        plt.scatter(source_mapped[:,0], source_mapped[:,1], c = source_labels, cmap = 'plasma',\
                     marker = '^', s = 10, label = hparams.simsuite, vmin=min_label, vmax=max_label)
        plt.scatter(target_mapped[:,0], target_mapped[:,1], c = target_labels, cmap = 'plasma',\
                     marker = 'o', s = 10, label = hparams.flip_suite(), vmin=min_label, vmax=max_label)

        ax = plt.gca()
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        ax.legend(loc = 'upper right', fontsize = 10)

        # Set colorbar to read true value for inferred parameter
        cbar = plt.colorbar()
        cbar.set_label('True value for \u03A9', rotation=270, labelpad = 20)

        # Add distance statistics to plot
        plt.text(0.05, 0.95, 'MMD = {:.2f}'.format(mmd), transform=ax.transAxes, fontsize=12, verticalalignment='top')

        if assessment:
            plt.savefig(dir + "{}_isomap_2D_assessment.png".format(hparams.name_model()), bbox_inches='tight', dpi=300)
        else:
            plt.savefig(dir + "{}_isomap_2D_epochs_{}.png".format(hparams.name_model(), epochs), bbox_inches='tight', dpi=300)
        plt.close(fig)

    elif n_components == 3:

        fig = plt.figure(figsize=(6,5))
        ax = plt.axes(projection='3d')
        if assessment:
            plt.title('Isomap 3D assessment')
        else:
            plt.title('Isomap 3D at epoch {}'.format(epochs))

        ax.scatter3D(source_mapped[:,0], source_mapped[:,1], source_mapped[:,2], c = labels[0], cmap = 'plasma',\
                        marker = '^', s = 15, label = hparams.simsuite, vmin=min_label, vmax=max_label)
        ax.scatter3D(target_mapped[:,0], target_mapped[:,1], target_mapped[:,2], c = labels[1], cmap = 'plasma',\
                        marker = 'o', s = 15, label = hparams.flip_suite(), vmin=min_label, vmax=max_label)
        
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.zaxis.set_ticks_position('none')

        ax.legend(loc = 'upper right', fontsize = 10)

        # Set colorbar to read true value for inferred parameter
        cbar = plt.colorbar()
        cbar.set_label('True value for \u03A9', rotation=270, labelpad = 20)

        # Add distance statistics to plot
        plt.text(0.05, 0.95, 'MMD = {:.2f}'.format(mmd), transform=ax.transAxes, fontsize=12, verticalalignment='top')

        if assessment:
            plt.savefig(dir + "{}_isomap_3D_assessment.png".format(hparams.name_model()), bbox_inches='tight', dpi=300)
        else:
            plt.savefig(dir + "{}_isomap_3D_epochs_{}.png".format(hparams.name_model(), epochs), bbox_inches='tight', dpi=300)
        plt.close(fig)

def remove_outliers(mapped, labels):
    """Remove outliers from mapped distributions to aid good visualization.

    Args:
        mapped (np.array): mapped distributions
        labels (list): list of labels for the source and target domains

    Returns:
        np.array: mapped distributions with outliers removed
        list: list of labels with outliers removed
    """
    # Remove outliers to aid good visualization and remove corresponding labels
    mapped = mapped[~np.isnan(mapped).any(axis=1)]
    labels = labels[~np.isnan(mapped).any(axis=1)]

    # Remove outliers at more than 5 std from the mean
    mean = np.mean(mapped, axis = 0)
    std = np.std(mapped, axis = 0)
    cond = np.all(np.abs(mapped - mean) < 5*std, axis = 1)
    mapped = mapped[cond]
    labels = labels[cond]

    return mapped, labels