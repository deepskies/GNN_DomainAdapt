import numpy as np

class EarlyStopping():

    """
    Implementation of early stopping.
    
    Attributes
    ----------
    patience (float) : number of consecutive worsening epochs allowed before early stopping
    tolerance (float) : minimum change in the monitored quantity to qualify as improvement
    mode (str, default = "min") : can be both "min" and "max", "max" needed for accuracy (in general, positive metrics)
    _best_value (float) : best value of the monitored quantity
    _n_epochs (int) : number of epochs passed since the beginning of training
    _n_worsening_epochs (int) : number of consecutive epochs in which the monitored quantity has worsened
    _best_epoch (int) : epoch in which the best value of the monitored quantity has been reached

    Methods
    -------
    __call__(value) : evaluates performance on validation set at the end of every epoch, eventually saving best parameters

    """

    def __init__ (self, patience, mode, tolerance):

        """
        Initialize EarlyStopping object.
        
        Parameters
        ----------
        patience (float) : number of consecutive worsening epochs allowed before early stopping
        tolerance (float) : minimum change in the monitored quantity to qualify as improvement
        mode (str, default = "min") : can be both "min" and "max", "max" needed for accuracy (in general, positive metrics)

        """

        self.patience = patience
        self.tolerance = tolerance
        self.mode = mode

        if self.mode == 'min':
            self._best_value = np.infty 
        elif self.mode == 'max':
            self._best_value = -np.infty
        self._n_epochs = 0
        self._n_worsening_epochs = 0

    def __call__(self, value):

        """
        Evaluates performance on validation set at the end of every epoch, eventually saving best parameters.

        Parameters
        ----------
        Returns (bool): True if training has to stop, False otherwise.

        """

        self._n_epochs += 1

        if (self.mode == "min" and value < self._best_value - self.tolerance) or (self.mode == "max" and value > self._best_value + self.tolerance):
            self._best_value = value
            self._n_worsening_epochs = 0
            self._best_epoch = self._n_epochs
        else:
            self._n_worsening_epochs += 1
            if self._n_worsening_epochs >= self.patience:
                return True
        
        return False