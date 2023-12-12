"""
Load the data from the simulations and create the dataset to train and test the GNN.

"""

import random

import h5py
import numpy as np
import scipy.spatial as SS
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .constants import BATCH_SIZE, BOX_SIZE, TEST_SIZE, VALID_SIZE, Nstar_th
from .utils.paths import datapath


# Normalize CAMELS parameters
def normalize_params(params):

    minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])
    maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])
    params = (params - minimum)/(maximum - minimum)
    return params

# Compute KDTree and get edges and edge features
def get_edges(pos, r_link, use_loops):

    # 1. Get edges

    # Create the KDTree and look for pairs within a distance r_link
    # Boxsize normalize to 1
    kd_tree = SS.KDTree(pos, leafsize=16, boxsize=1.0001)
    edge_index = kd_tree.query_pairs(r=r_link, output_type="ndarray")

    # Add reverse pairs
    reversepairs = np.zeros((edge_index.shape[0],2))
    for i, pair in enumerate(edge_index):
        reversepairs[i] = np.array([pair[1], pair[0]])
    edge_index = np.append(edge_index, reversepairs, 0)

    edge_index = edge_index.astype(int)

    # Write in pytorch-geometric format
    edge_index = edge_index.T
    num_pairs = edge_index.shape[1]

    # 2. Get edge attributes

    row, col = edge_index
    diff = pos[row]-pos[col]

    # Take into account periodic boundary conditions, correcting the distances
    for i, pos_i in enumerate(diff):
        for j, coord in enumerate(pos_i):
            if coord > r_link:
                diff[i,j] -= 1.  # Boxsize normalize to 1
            elif -coord > r_link:
                diff[i,j] += 1.  # Boxsize normalize to 1

    # Get translational and rotational invariant features
    
    # Distance
    dist = np.linalg.norm(diff, axis=1)
    
    # Centroid of galaxy catalogue
    centroid = np.mean(pos,axis=0)

    #Vectors of node and neighbor
    row = (pos[row] - centroid)
    col = (pos[col] - centroid)

   # Take into account periodic boundary conditions: row and col
    for i, pos_i in enumerate(row):
        for j, coord in enumerate(pos_i):
            if coord > 0.5:
                row[i,j] -= 1.  # Boxsize normalize to 1
                
            elif -coord > 0.5:
                row[i,j] += 1.  # Boxsize normalize to 1                                                

    for i, pos_i in enumerate(col):
        for j, coord in enumerate(pos_i):
            if coord > 0.5:
                col[i,j] -= 1.  # Boxsize normalize to 1
                
            elif -coord > 0.5:
                col[i,j] += 1.  # Boxsize normalize to 1
                
    # Normalizing
    unitrow = row/np.linalg.norm(row, axis = 1).reshape(-1, 1)
    unitcol = col/np.linalg.norm(col, axis = 1).reshape(-1, 1)
    unitdiff = diff/dist.reshape(-1,1)
    
    # Dot products between unit vectors
    cos1 = np.array([np.dot(unitrow[i,:].T,unitcol[i,:]) for i in range(num_pairs)])
    cos2 = np.array([np.dot(unitrow[i,:].T,unitdiff[i,:]) for i in range(num_pairs)])

    # Normalize distance by linking radius
    dist /= r_link

    # Concatenate to get all edge attributes
    edge_attr = np.concatenate([dist.reshape(-1,1), cos1.reshape(-1,1), cos2.reshape(-1,1)], axis=1)

    # Add loops
    if use_loops:
        loops = np.zeros((2,pos.shape[0]),dtype=int)
        atrloops = np.zeros((pos.shape[0],3))
        for i, posit in enumerate(pos):
            loops[0,i], loops[1,i] = i, i
            atrloops[i,0], atrloops[i,1], atrloops[i,2] = 0., 1., 0.
        edge_index = np.append(edge_index, loops, 1)
        edge_attr = np.append(edge_attr, atrloops, 0)
    edge_index = edge_index.astype(int)

    return edge_index, edge_attr

# Routine to create a cosmic graph from a galaxy catalogue
# simsuite: the simulation suite
# simnumber: number of simulation
# param_file: file with the value of the cosmological + astrophysical parameters
# hparams: hyperparameters class
def sim_graph(simsuite, simnumber, param_file, hparams):

    # Get some hyperparameters
    simsuite = simsuite
    simset,r_link,only_positions,pred_params = hparams.simset,hparams.r_link,hparams.only_positions,hparams.pred_params

    # Name of the galaxy catalogue
    simpath = datapath + simsuite + "_" + simset + "_"
    catalogue_path = simpath + str(simnumber) + "_fof_subhalo_tab_0"+hparams.snap+".hdf5"

    # Read the catalogue
    catalogue = h5py.File(catalogue_path, 'r')
    pos   = catalogue['/Subhalo/SubhaloPos'][:]/BOX_SIZE
    Mstar = catalogue['/Subhalo/SubhaloMassType'][:,4] #Msun/h
    Rstar = catalogue["Subhalo/SubhaloHalfmassRadType"][:,4]
    Metal = catalogue["Subhalo/SubhaloStarMetallicity"][:]
    Vmax = catalogue["Subhalo/SubhaloVmax"][:]
    Nstar = catalogue['/Subhalo/SubhaloLenType'][:,4]       #number of stars
    catalogue.close()

    # Some simulations are slightly outside the box, correct it
    pos[np.where(pos<0.0)]+=1.0
    pos[np.where(pos>1.0)]-=1.0

    # Select only galaxies with more than Nstar_th star particles
    indexes = np.where(Nstar>Nstar_th)[0]
    pos     = pos[indexes]
    Mstar   = Mstar[indexes]
    Rstar   = Rstar[indexes]
    Metal   = Metal[indexes]
    Vmax   = Vmax[indexes]

    # Get the output to be predicted by the GNN,he cosmo parameters
    # Read the value of the cosmological & astrophysical parameters
    paramsfile = np.loadtxt(param_file, dtype=str)
    params = np.array(paramsfile[simnumber,1:-1],dtype=np.float32)
    params = normalize_params(params)
    params = params[:pred_params]   # Consider only the first parameters, up to pred_params
    y = np.reshape(params, (1,params.shape[0]))

    # Number of galaxies as global feature
    u = np.log10(pos.shape[0]).reshape(1,1)

    Mstar = np.log10(1.+ Mstar)
    Rstar = np.log10(1.+ Rstar)
    Metal = np.log10(1.+ Metal)
    Vmax = np.log10(1. + Vmax)

    # Node features
    tab = np.column_stack((Mstar, Rstar, Metal, Vmax))
    #tab = Vmax.reshape(-1,1)       # For using only Vmax
    x = torch.tensor(tab, dtype=torch.float32)

    # Use loops if node features are considered only
    if only_positions:
        tab = np.zeros_like(pos[:,:1])   # Node features not really used
        use_loops = False
    else:
        use_loops = True

    # Get edges and edge features
    edge_index, edge_attr = get_edges(pos, r_link, use_loops)

    # Construct the graph
    graph = Data(x=x,
                 y=torch.tensor(y, dtype=torch.float32),
                 u=torch.tensor(u, dtype=torch.float32),
                 edge_index=torch.tensor(edge_index, dtype=torch.long),
                 edge_attr=torch.tensor(edge_attr, dtype=torch.float32))

    return graph


# Split training and validation sets
def split_datasets(dataset):

    random.shuffle(dataset)

    num_train = len(dataset)
    split_valid = int(np.floor(VALID_SIZE * num_train))
    split_test = split_valid + int(np.floor(TEST_SIZE * num_train))

    train_dataset = dataset[split_test:]
    valid_dataset = dataset[:split_valid]
    test_dataset = dataset[split_valid:split_test]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, valid_loader, test_loader

######################################################################################

# Main routine to load data and create the dataset
def create_dataset(hparams, verbose = True):
    """Create the dataset to train and test the GNN.
    There are two datasets, one for each simulation suite.

    Args:
        hparams (HyperParameters): Hyperparameters of GNN, which include some characteristics of dataset.

    Returns:
        dict: dictionary with either one or two datasets, where the key is the name of the simulation
    """

    datasets = {}
    datasets[hparams.simsuite] = []
    datasets[hparams.flip_suite()] = []

    for name, dataset in datasets.items():
        param_file = datapath + "CosmoAstroSeed_params_" + name + ".txt"
        for simnumber in range(hparams.n_sims):
            dataset.append(sim_graph(name,simnumber,param_file,hparams))
        if verbose: 
            print("Loaded {} simulations from {}".format(len(dataset), name))

    gals = {}
    for name, dataset in datasets.items():
        gals[name] = np.array([graph.x.shape[0] for graph in datasets[name]])
        if verbose:
            print("Simulation suite {} statistics:".format(name))
            print("Total of galaxies", gals[name].sum(0), "Mean of", gals[name].mean(0),"per simulation, Std of", gals[name].std(0))

    return datasets
