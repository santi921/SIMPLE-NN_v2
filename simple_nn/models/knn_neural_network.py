from __future__ import division

import torch
from torch.nn import Linear

import numpy as np; npl = np.linalg
from scipy.linalg import block_diag
from time import time
import pickle
import shutil

from simple_nn.models import weight_initializers


class FCNDict(torch.nn.Module): # parser object for NN later
    def __init__(self, nets):
        super(FCNDict, self).__init__()
        self.nets = torch.nn.ModuleDict(nets)
        self.keys = self.nets.keys() # linear component to nn

    def forward(self, x):  
        assert [item for item in self.nets.keys()].sort() == [item for item in x.keys()].sort()
        res = {}
        for key in x:
            res[key] = self.nets[key](x[key])

        return res

    def write_lammps_potential(self, filename, inputs, scale_factor=None, pca=None):
        # TODO: get the parameter info from initial batch generting processs
        atom_type_str = ' '.join(inputs['atom_types'])

        FIL = open(filename, 'w')
        FIL.write('ELEM_LIST ' + atom_type_str + '\n\n')

        for item in inputs['atom_types']:
            params = list()
            with open(inputs['params'][item]) as fil:
                for line in fil:
                    tmp = line.split()
                    params += [list(map(float, tmp))]
            params = np.array(params)

            FIL.write('POT {} {}\n'.format(item, np.max(params[:,3])))
            FIL.write('SYM {}\n'.format(len(params)))

            for ctem in params:
                tmp_types = inputs['atom_types'][int(ctem[1]) - 1]
                if int(ctem[0]) > 3:
                    tmp_types += ' {}'.format(inputs['atom_types'][int(ctem[2])-1])
                if len(ctem) != 7:
                    raise ValueError("params file must have lines with 7 columns.")

                FIL.write('{} {} {} {} {} {}\n'.\
                    format(int(ctem[0]), ctem[3], ctem[4], ctem[5], ctem[6], tmp_types))

            if scale_factor is None:
                with open(inputs['params'][item], 'r') as f:
                    tmp = f.readlines()
                input_dim = len(tmp) #open params read input number of symmetry functions
                FIL.write('scale1 {}\n'.format(' '.join(np.zeros(input_dim).astype(np.str))))
                FIL.write('scale2 {}\n'.format(' '.join(np.ones(input_dim).astype(np.str))))
            else:
                FIL.write('scale1 {}\n'.format(' '.join(scale_factor[item][0].cpu().numpy().astype(np.str))))
                FIL.write('scale2 {}\n'.format(' '.join(scale_factor[item][1].cpu().numpy().astype(np.str))))

            # An extra linear layer is used for PCA transformation.
            nodes   = list()
            weights = list()
            biases  = list()
            for n, i in self.nets[item].lin.named_modules():
                if 'lin' in n:
                    nodes.append(i.weight.size(0))
                    weights.append(i.weight.detach().cpu().numpy())
                    biases.append(i.bias.detach().cpu().numpy())
            nlayers = len(nodes)
            if pca is not None:
                nodes = [pca[item][0].cpu().numpy().shape[1]] + nodes
                joffset = 1
            else:
                joffset = 0
            FIL.write('NET {} {}\n'.format(len(nodes)-1, ' '.join(map(str, nodes))))

            # PCA transformation layer.
            if pca is not None:
                FIL.write('LAYER 0 linear PCA\n')
                pca_mat = np.copy(pca[item][0].cpu().numpy())
                pca_mean = np.copy(pca[item][2].cpu().numpy())
                if inputs['preprocessing']['min_whiten_level'] is not None:
                    pca_mat /= pca[item][1].cpu().numpy().reshape([1, -1])
                    pca_mean /= pca[item][1].cpu().numpy()

                for k in range(pca[item][0].cpu().numpy().shape[1]):
                    FIL.write('w{} {}\n'.format(k, ' '.join(pca_mat[:,k].astype(np.str))))
                    FIL.write('b{} {}\n'.format(k, -pca_mean[k]))

            for j in range(nlayers):
                # FIXME: add activation function type if new activation is added
                if j == nlayers-1:
                    acti = 'linear'
                else:
                    acti = inputs['neural_network']['acti_func']

                FIL.write('LAYER {} {}\n'.format(j+joffset, acti))

                for k in range(nodes[j + joffset]):
                    FIL.write('w{} {}\n'.format(k, ' '.join(weights[j][k,:].astype(np.str))))
                    FIL.write('b{} {}\n'.format(k, biases[j][k]))

            FIL.write('\n')
        FIL.close()

class FCN(torch.nn.Module):
    def __init__(self, dim_input, dim_hidden, acti_func='relu', dropout=0.0):
        super(FCN, self).__init__()

        self.lin = torch.nn.Sequential()

        dim_in = dim_input
        for i, hn in enumerate(dim_hidden): # replace this with kalman and profit?
            self.lin.add_module(f'lin_{i}', torch.nn.Linear(dim_in, hn))
            #if batch_norm:
            #    seq.add_module(torch.nn.BatchNorm1d(hn))
            dim_in = hn
            if acti_func == 'sigmoid':
                self.lin.add_module(f'sigmoid_{i}', torch.nn.Sigmoid())
            elif acti_func == 'tanh':
                self.lin.add_module(f'tanh_{i}', torch.nn.Tanh())
            elif acti_func == 'relu':
                self.lin.add_module(f'relu_{i}', torch.nn.ReLU())
            elif acti_func == 'selu':
                self.lin.add_module(f'tanh_{i}', torch.nn.SELU())
            elif acti_func == 'swish':
                self.lin.add_module(f'swish_{i}', swish())
            else:
                assert False

            if dropout:
                self.lin.add_module(f'drop_{i}', torch.nn.Dropout(p=dropout))

        self.lin.add_module(f'lin_{i+1}', torch.nn.Linear(hn, 1))


        # Initial synapse weight matrices
        sprW = 5
        sprW = np.float64(sprW)
        self.W = [sprW*(2*np.random.sample((nl, nu+1))-1),
                  sprW*(2*np.random.sample((ny, nl+1))-1)]
        self.nW = sum(map(np.size, self.W))
        self.P = None

        # Function for pushing signals through a synapse with bias
        self._affine_dot = lambda W, V: np.dot(np.atleast_1d(V), W[:, :-1].T) + W[:, -1]

        # Function for computing the RMS error of the current fit to some data set
        self.compute_rms = lambda U, Y: np.sqrt(np.mean(np.square(Y - self.feedforward(U))))

    def save(self, filename):
        """
        Saves the current NN to a file with the given string filename.
        """
        if not isinstance(filename, str):
            raise ValueError("The filename must be a string.")
        if filename[-4:] != '.knn':
            filename = filename + '.knn'
        with open(filename, 'wb') as output:
            pickle.dump((self.W, self.neuron, self.P), output, pickle.HIGHEST_PROTOCOL)


    def forward(self, x):
        return self.lin(x)



# loader helper function
#def load_knn(filename):
#    """
#    Loads a stored KNN object saved with the string filename.
#    Returns the loaded object.
#    """
#    if not isinstance(filename, str):
#        raise ValueError("The filename must be a string.")
#    if filename[-4:] != '.knn':
#        filename = filename + '.knn'
#    with open(filename, 'rb') as input:
#        W, neuron, P = pickle.load(input)
#    obj = KNN(W[0].shape[1]-1, W[1].shape[0], W[0].shape[0], neuron)
#    obj.W, obj.P = W, P
#    return obj



class KNN:
    """
    Class for a feedforward neural network (NN). Currently only handles 1 hidden-layer,
    is always fully-connected, and uses the same activation function type for every neuron.
    The NN can be trained by extended kalman filter (EKF) or stochastic gradient descent (SGD).
    Use the train function to train the NN, the feedforward function to compute the NN output,
    and the classify function to round a feedforward to the nearest class values. A save function
    is also provided to store a KNN object in the working directory.
    """
    def __init__(self, nu, ny, nl, neuron, sprW=5):
        """
            nu: dimensionality of input; positive integer
            ny: dimensionality of output; positive integer
            nl: number of hidden-layer neurons; positive integer
        neuron: activation function type; 'logistic', 'tanh', or 'relu'
          sprW: spread of initial randomly sampled synapse weights; float scalar
        """
        '''
        # Function dimensionalities
        self.nu = int(nu)
        self.ny = int(ny)
        self.nl = int(nl)

        # Neuron type
        if neuron == 'logistic':
            self.sig = lambda V: (1 + np.exp(-V))**-1
            self.dsig = lambda sigV: sigV * (1 - sigV)
        elif neuron == 'tanh':
            self.sig = lambda V: np.tanh(V)
            self.dsig = lambda sigV: 1 - sigV**2
        elif neuron == 'relu':
            self.sig = lambda V: np.clip(V, 0, np.inf)
            self.dsig = lambda sigV: np.float64(sigV > 0)
        else:
            raise ValueError("The neuron argument must be 'logistic', 'tanh', or 'relu'.")
        self.neuron = neuron
        '''

        #self.sig = lambda V: np.clip(V, 0, np.inf)
        #self.dsig = lambda sigV: np.float64(sigV > 0)

        # Initial synapse weight matrices
        sprW = np.float64(sprW)
        self.W = [sprW*(2*np.random.sample((nl, nu+1))-1),
                  sprW*(2*np.random.sample((ny, nl+1))-1)]
        self.nW = sum(map(np.size, self.W))
        self.P = None

        # Function for pushing signals through a synapse with bias
        self._affine_dot = lambda W, V: np.dot(np.atleast_1d(V), W[:, :-1].T) + W[:, -1]

        # Function for computing the RMS error of the current fit to some data set
        self.compute_rms = lambda U, Y: np.sqrt(np.mean(np.square(Y - self.feedforward(U))))

    '''
    def save(self, filename):
        """
        Saves the current NN to a file with the given string filename.
        """
        if not isinstance(filename, str):
            raise ValueError("The filename must be a string.")
        if filename[-4:] != '.knn':
            filename = filename + '.knn'
        with open(filename, 'wb') as output:
            pickle.dump((self.W, self.neuron, self.P), output, pickle.HIGHEST_PROTOCOL)
    '''

    def feedforward(self, U, get_l=False):
        """
        Feeds forward an (m by nu) array of inputs U through the NN.
        Returns the associated (m by ny) output matrix, and optionally
        the intermediate activations l.
        """
        U = np.float64(U)
        if U.ndim == 1 and len(U) > self.nu: U = U[:, np.newaxis]
        l = self.sig(self._affine_dot(self.W[0], U))
        h = self._affine_dot(self.W[1], l)
        if get_l: return h, l
        return h


    def classify(self, U, high, low=0):
        """
        Feeds forward an (m by nu) array of inputs U through the NN.
        For each associated output, the closest integer between high
        and low is returned as a (m by ny) classification matrix.
        Basically, your training data should be (u, int_between_high_low).
        """
        return np.int64(np.clip(np.round(self.feedforward(U), 0), low, high))


    def train(self, nepochs, U, Y, method, P=None, Q=None, R=None, step=1, dtol=-1, dslew=1, pulse_T=-1):
        """
        nepochs: number of epochs (presentations of the training data); integer
              U: input training data; float array m samples by nu inputs
              Y: output training data; float array m samples by ny outputs
         method: extended kalman filter ('ekf') or stochastic gradient descent ('sgd')
              P: initial weight covariance for ekf; float scalar or (nW by nW) posdef array
              Q: process covariance for ekf; float scalar or (nW by nW) semiposdef array
              R: data covariance for ekf; float scalar or (ny by ny) posdef array
           step: step-size scaling; float scalar
           dtol: finish when RMS error avg change is <dtol (or nepochs exceeded); float scalar
          dslew: how many deltas over which to examine average RMS change; integer
        pulse_T: number of seconds between displaying current training status; float
        If method is 'sgd' then P, Q, and R are unused, so carefully choose step.
        If method is 'ekf' then step=1 is "optimal", R must be specified, and:
            P is None: P = self.P if self.P has been created by previous training
            Q is None: Q = 0
        If P, Q, or R are given as scalars, they will scale an identity matrix.
        Set pulse_T to -1 (default) to suppress training status display.
        Returns a list of the RMS errors at every epoch and a list of the covariance traces
        at every iteration. The covariance trace list will be empty if using sgd.
        """
        # Verify data
        U = np.float64(U)
        Y = np.float64(Y)
        if len(U) != len(Y):
            raise ValueError("Number of input data points must match number of output data points.")
        if (U.ndim == 1 and self.nu != 1) or (U.ndim != 1 and U.shape[-1] != self.nu):
            raise ValueError("Shape of U must be (m by nu).")
        if (Y.ndim == 1 and self.ny != 1) or (Y.ndim != 1 and Y.shape[-1] != self.ny):
            raise ValueError("Shape of Y must be (m by ny).")
        if Y.ndim == 1 and len(Y) > self.ny: Y = Y[:, np.newaxis]

        # Set-up
        if method == 'ekf':
            self.update = self._ekf

            if P is None:
                if self.P is None:
                    raise ValueError("Initial P not specified.")
            elif np.isscalar(P):
                self.P = P*np.eye(self.nW)
            else:
                if np.shape(P) != (self.nW, self.nW):
                    raise ValueError("P must be a float scalar or (nW by nW) array.")
                self.P = np.float64(P)

            if Q is None:
                self.Q = np.zeros((self.nW, self.nW))
            elif np.isscalar(Q):
                self.Q = Q*np.eye(self.nW)
            else:
                if np.shape(Q) != (self.nW, self.nW):
                    raise ValueError("Q must be a float scalar or (nW by nW) array.")
                self.Q = np.float64(Q)
            if np.any(self.Q): self.Q_nonzero = True
            else: self.Q_nonzero = False

            if R is None:
                raise ValueError("R must be specified for EKF training.")
            elif np.isscalar(R):
                self.R = R*np.eye(self.ny)
            else:
                if np.shape(R) != (self.ny, self.ny):
                    raise ValueError("R must be a float scalar or (ny by ny) array.")
                self.R = np.float64(R)
            if npl.matrix_rank(self.R) != len(self.R):
                raise ValueError("R must be positive definite.")

        elif method == 'sgd':
            self.update = self._sgd
        else:
            raise ValueError("The method argument must be either 'ekf' or 'sgd'.")
        last_pulse = 0
        RMS = []
        trcov = []

        # Shuffle data between epochs
        print("Training...")
        for epoch in range(nepochs):
            rand_idx = np.random.permutation(len(U))
            U_shuffled = U[rand_idx]
            Y_shuffled = Y[rand_idx]
            RMS.append(self.compute_rms(U, Y))

            # Check for convergence
            if len(RMS) > dslew and abs(RMS[-1] - RMS[-1-dslew])/dslew < dtol:
                print("\nConverged after {} epochs!\n\n".format(epoch+1))
                return RMS, trcov

            # Train
            for i, (u, y) in enumerate(zip(U_shuffled, Y_shuffled)):

                # Forward propagation
                h, l = self.feedforward(u, get_l=True)

                # Do the learning
                self.update(u, y, h, l, step)
                if method == 'ekf': trcov.append(np.trace(self.P))

                # Heartbeat
                if (pulse_T >= 0 and time()-last_pulse > pulse_T) or (epoch == nepochs-1 and i == len(U)-1):
                    print("------------------")
                    print("  Epoch: {}%".format(int(100*(epoch+1)/nepochs)))
                    print("   Iter: {}%".format(int(100*(i+1)/len(U))))
                    print("   RMSE: {}".format(np.round(RMS[-1], 6)))
                    if method == 'ekf': print("tr(Cov): {}".format(np.round(trcov[-1], 6)))
                    print("------------------")
                    last_pulse = time()
        print("\nTraining complete!\n\n")
        RMS.append(self.compute_rms(U, Y))
        return RMS, trcov


    def _ekf(self, u, y, h, l, step):

        # Compute NN jacobian
        D = (self.W[1][:, :-1]*self.dsig(l)).flatten()
        H = np.hstack((np.hstack((np.outer(D, u), D[:, np.newaxis])).reshape(self.ny, self.W[0].size),
                       block_diag(*np.tile(np.concatenate((l, [1])), self.ny).reshape(self.ny, self.nl+1))))

        # Kalman gain
        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T).dot(npl.inv(S))

        # Update weight estimates and covariance
        dW = step*K.dot(y-h)
        self.W[0] = self.W[0] + dW[:self.W[0].size].reshape(self.W[0].shape)
        self.W[1] = self.W[1] + dW[self.W[0].size:].reshape(self.W[1].shape)
        self.P = self.P - np.dot(K, H.dot(self.P))
        if self.Q_nonzero: self.P = self.P + self.Q


    def _sgd(self, u, y, h, l, step):
        e = h - y
        self.W[1] = self.W[1] - step*np.hstack((np.outer(e, l), e[:, np.newaxis]))
        D = (e.dot(self.W[1][:, :-1])*self.dsig(l)).flatten()
        self.W[0] = self.W[0] - step*np.hstack((np.outer(D, u), D[:, np.newaxis]))






def _initialize_model_and_weights(inputs, logfile, device):
    logfile.write(f"{device} is used in model.\n")

    model = {}
    for element in inputs['atom_types']:
        hidden_layer_nodes = [int(nodes) for nodes in inputs['neural_network']['nodes'].split('-')]

        # change if calculating input_nodes method is changed
        with open(inputs['params'][element], 'r') as f:
            tmp_symf = f.readlines()
            input_nodes = len(tmp_symf)

        model[element] = FCN(input_nodes, hidden_layer_nodes,\
            acti_func=inputs['neural_network']['acti_func'],
            dropout=inputs['neural_network']['dropout'])
        #model[element] = KalmanNN(input_nodes, hidden_layer_nodes,\
        #    acti_func=inputs['neural_network']['acti_func'],
        #    dropout=inputs['neural_network']['dropout'])
        

        weights_initialize_log = weight_initializers._initialize_weights(inputs, logfile, model[element])

    model = FCNDict(model) #Make full model with elementized dictionary model, nn  obj only creates model in pytorch
    model.to(device=device)

    return model

def read_lammps_potential(filename):
    def _read_until(fil, stop_tag):
        while True:
            line = fil.readline()
            if stop_tag in line:
                break

        return line

    shutil.copy2(filename, 'potential_read')

    weights = dict()
    with open(filename) as fil:
        atom_types = fil.readline().replace('\n', '').split()[1:]
        for item in atom_types:
            weights[item] = dict()

            dims = list()
            dims.append(int(_read_until(fil, 'SYM').split()[1]))

            hidden_to_out = map(lambda x: int(x), _read_until(fil, 'NET').split()[2:])
            dims += hidden_to_out

            num_weights = len(dims) - 1

            tmp_idx = 0
            for j in range(num_weights):
                weights[item][f'lin_{tmp_idx}'] = dict()
                tmp_weights = np.zeros([dims[j], dims[j+1]])
                tmp_bias = np.zeros([dims[j+1]])

                # Since PCA will be dealt separately, skip PCA layer.
                skip = True if fil.readline().split()[-1] == 'PCA' else False
                for k in range(dims[j+1]):
                    tmp_weights[:,k] = list(map(lambda x: float(x), fil.readline().split()[1:]))
                    tmp_bias[k] = float(fil.readline().split()[1])

                weights[item][f'lin_{tmp_idx}']['weight'] = np.copy(tmp_weights)
                weights[item][f'lin_{tmp_idx}']['bias']  = np.copy(tmp_bias)
                if skip:
                    continue
                else:
                    tmp_idx += 1
    return weights
