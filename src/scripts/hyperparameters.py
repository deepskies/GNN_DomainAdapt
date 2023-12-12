""" 
Definition of the HyperParameters class, which contains all the hyperparameters of the model.
It also contains hyperparameters useful for constructing the graph, such as the linking radius.

"""

class HyperParameters():
    def __init__(   self,
                    simsuite,           # Simulation suite, choose between "IllustrisTNG" and "SIMBA"
                    simset,             # Simulation set, choose between "CV" and "LH"
                    n_sims,             # Number of simulations considered, maximum 27 for CV and 1000 for LH 
                    domain_adapt,       # Domain Adaptation type
                    training,           # If training, set to True, otherwise loads a pretrained model and tests it
                    pred_params,        # Number of cosmo/astro params to be predicted, starting from Omega_m, sigma_8, etc.
                    only_positions,     # 1 for using only positions as features, 0 for using additional galactic features
                    snap,               # Snapshot of the simulation, indicating redshift 4: z=3, 10: z=2, 14: z=1.5, 18: z=1, 24: z=0.5, 33: z=0
                    da_loss_fraction,   # Fraction of the total loss that is the MMD loss
                    r_link,             # Linking radius to build the graph
                    n_layers,           # Number of graph layers
                    hidden_channels,    # Hidden channels
                    n_epochs,           # Number of epochs
                    learning_rate,      # Learning rate
                    weight_decay,       # Weight decay
                    weight_da           # Domain adaptation weight
                ):
        # Set non optimizable hyperparameters (construction choices)
        self.simsuite = simsuite
        self.simset = simset
        self.n_sims = n_sims
        self.domain_adapt = domain_adapt
        self.training = training
        self.pred_params = pred_params
        self.only_positions = only_positions
        self.snap = snap
        self.da_loss_fraction = da_loss_fraction

        # Set optimizable hyperparameters
        self.r_link = r_link
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.weight_da = weight_da

    def __str__(self):
        """Print the hyperparameters.

        Returns:
            str: String containing the hyperparameters.
        """
        return "Hyperparameters:\n"+\
            "Simulation suite: "+self.simsuite+"\n"+\
            "Simulation set: "+self.simset+"\n"+\
            "Number of simulations: "+str(self.n_sims)+"\n"+\
            "Domain adaptation: "+self.domain_adapt+"\n"+\
            "Training: "+str(self.training)+"\n"+\
            "Number of predicted parameters: "+str(self.pred_params)+"\n"+\
            "Only positions: "+str(self.only_positions)+"\n"+\
            "Snapshot: "+self.snap+"\n"+\
            "DA loss fraction: "+str(self.da_loss_fraction)+"\n"+\
            "Linking radius: "+str(self.r_link)+"\n"+\
            "Number of graph layers: "+str(self.n_layers)+"\n"+\
            "Hidden channels: "+str(self.hidden_channels)+"\n"+\
            "Number of epochs: "+str(self.n_epochs)+"\n"+\
            "Learning rate: "+str(self.learning_rate)+"\n"+\
            "Weight decay: "+str(self.weight_decay)+"\n"+\
            "Domain adaptation weight: "+str(self.weight_da)+"\n"


    def name_model(self):
        """Generate the name of the model for file saving.

        Returns:
            str: Name of the model.
        """
        return self.simsuite+"_"+self.simset+"_"+self.domain_adapt+"_FR_"+str(self.da_loss_fraction)+"_onlypos_"+str(self.only_positions)+\
            "_lr_{:.2e}_weight-da_{:.2e}_weightdecay_{:.2e}_layers_{:d}_rlink_{:.2e}_channels_{:d}_epochs_{:d}".format\
            (self.learning_rate, self.weight_da, self.weight_decay, self.n_layers, self.r_link, self.hidden_channels, self.n_epochs)
    
    def flip_suite(self):
        """Return the other CAMELS simulation suite.

        Returns:
            str: Other CAMELS simulation suite.
        """
        if self.simsuite=="IllustrisTNG":
            new_simsuite = "SIMBA"
        elif self.simsuite=="SIMBA":
            new_simsuite = "IllustrisTNG"
        return new_simsuite
