import numpy as np
import numpy.matlib
import math
import matplotlib.pyplot as plt
import csv
import time
from scipy.io import loadmat

'''A multilayer perceptron class that can hold any number of hidden layers
   written by Luca Del Basso
   See the acompanying report for more details on the assignment
   '''

class MLP(object):
    """A multilayey perceptron class
    """
    
    def __init__(self, n_inputs, h_layers, n_outputs, n_epoch, data_set, softmax=False,ema=False):
        """Constructor for the MLP. Takes the number of inputs,
            a variable number of hidden layers, and number of outputs
        Args:
            n_inputs (int): Number of inputs
            h_layers (list(int)): A list of ints for the hidden layers (can be empty [] for a SLP)
            n_outputs (int): Number of outputs
            n_epoch (int): Number of epochs
            data_set (ndarray): the whole dataset (training and test)
            softmax (bool): to run softmax on the output layer (instead of ReLU)
            ema(bool): to calcuate the mean weight update over time (only used for SLP)
        """
        
        self.n_inputs = n_inputs
        self.h_layers = h_layers
        self.n_outputs = n_outputs
        self.n_epoch = n_epoch
        self.data_set = data_set
        self.softmax = softmax
        self.ema = ema

        
        layers = [n_inputs] + h_layers + [n_outputs]
        n_layers = len(layers)
        
        if len(layers)>2 and self.ema:
            print("ema only runs on a single layer perceptron")
            print("removing hidden layers")
            layers = [layers[0],layers[-1]]

        self.weights = [ np.random.uniform(-1,1,(layers[i+1],layers[i])) * np.sqrt(1 / (layers[i]))
                               for i in range(n_layers-1) ]

        self.bias =[ np.zeros((layers[i+1],))
                           for i in range(n_layers-1)]

        self.errors = np.zeros((n_epoch,))       
        
    def forward_propagate(self, inputs,batch_s):
        """Computes forward propagation of the network based on input signals.
        Args:
            inputs (ndarray): Input signals
        Returns:
            activations (ndarray): Output values
        """
        # the input layer activation is just the input itself
        activations = inputs
        h_outs = []
        x_outs = [inputs]
        # iterate through the network layers
        for i in range(len(self.weights)):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.matmul(self.weights[i],activations) + self.bias[i][:,None]
            h_outs.append(net_inputs)
            # apply sigmoid activation function
            if self.softmax:
                if i == len(self.weights)-1:#softmax on output
                    a = softmax(net_inputs)
                    activations = softmax(net_inputs)
                else:
                    activations = np.maximum(0.0,net_inputs)
            else:
                activations = np.maximum(0.0,net_inputs)

            x_outs.append(activations)
        # return output layer activation
        return x_outs,h_outs


    def back_propagation(self,outputs,h_out,e_n,eta,B):
        """Computes backward propagation of the network.
        Args:
            outputs (ndarray): f(h) values from every layer
            h_out (ndarray): h values from every layer
            e_n   (ndarray): the error signal
            eta   (float): the learning rate
            B     (int): the batch size
        Returns:
            dws (list(ndarray)): delta weights
            dbs (list(ndarray)): delta bias

        """
        bp = len(self.weights)
        out = outputs[-1]

        dws = [ np.zeros(w.shape) for w in self.weights ]
        dbs = [ np.zeros(b.shape) for b in self.bias ]

        last_delta = None
        for i in range(bp,0,-1):
            if i == bp:
                #perform softmax on the output layer
                if self.softmax:
                    delta = e_n 
                else:
                    delta = e_n * d_abs(h_out[-1]) 
            else:
                delta = d_abs(h_out[i-1])* np.matmul(self.weights[i].T,last_delta)
            last_delta = delta       
        
            dws[i-1] = (eta/B) * np.matmul(delta,outputs[i-1].T)
            dbs[i-1] = (eta/B) * np.sum(delta,axis=1)
        return dws,dbs

        
    def train(self, x_data,y_train, n_epoch,b_size,eta, comparison_set, string ="",L1=False,lambD = 0.00001,tau=0.0):
        """Trains the model.
        Args:
            x_data (ndarray): the training dataset
            y_train (ndarray): labels for the training dataset
            n_epoch  (int): number of epochs to run
            b_size (int): the batch size
            eta   (float): the learning rate
            comparison_set (list(ndarray)): the validation/test set and the validation/test labels respectively
            string(int): an optional string to print when running experiments 
            L1 (boolean): enable or disable the L1 normalization
            lambD (float): lambda values for L1
            tau (float): for mean averaging
        Returns:
            errors (ndarray): the error of the training stage
            accuracy:  the accuracy of the training stage

        """
        errors = np.zeros((n_epoch,))
        n_samples = x_data.shape[0]
        batch_size = b_size 
        n_batches = int(math.ceil(n_samples/batch_size))
        accuracy = np.zeros((n_epoch,))
        average = np.zeros(self.weights[0].shape)
        means = np.zeros((n_batches,))
        tau = tau
        lamb = lambD
        for i in range(n_epoch):
            t = time.time()
          
            shuffled_idxs = np.random.permutation(n_samples)
                      
            for batch in range(n_batches):                
                    
                idxs = shuffled_idxs[batch*batch_size :batch*batch_size+batch_size]
                x0_batch = x_data[idxs].T
                
                desired_out =y_train[idxs].T
                
                ##neural activation for each layer
                #return all layer outputs for back prop
                x_outs,h_outs = self.forward_propagate(x0_batch,batch_size)

                e_n = desired_out - x_outs[-1] #x_out[-1] is the final output
                
                dws,dbs = self.back_propagation(x_outs,h_outs,e_n,eta,batch_size)
                
                #L1 normalization, if enabled
                l1,d_l1 = 0,0
                if L1:
                    l1 = lamb *sum([np.sum(np.abs(w)) for w in self.weights])
                    d_l1 =  [d_abs(w) for w in self.weights]
                    
                if self.softmax:
                    errors[i] += np.sum(-desired_out*np.log(x_outs[-1]))/batch_size
                else:#MSE
                    errors[i] += (1/(2*batch_size))*np.sum(np.square(e_n)) + (l1/batch_size)

                for k in range(len(self.weights)):
                    self.weights[k] += dws[k]
                    self.bias[k]+= dbs[k]
                    if L1:
                        self.weights[k] -=(eta/batch_size) *lamb*d_l1[k]
                        
                    if batch == 0 and self.ema:##exponential mean average
                        average += dws[0]
                        means[batch] += np.mean(average)
                    elif self.ema:
                        average = average*(1-tau) + tau*(dws[0])
                        means[batch] += np.mean(average)
            #compare to test or validation set
            outs,hs = self.forward_propagate(comparison_set[0].T,comparison_set[0].T.shape[1])
            a_out = np.argmax(outs[-1],axis=0)
            e_out =  np.argmax(comparison_set[1].T,axis=0)
            a = a_out==e_out
            accuracy[i] = np.sum(a)/len(a) * 100
            print(string+ f"Epoch: {i+1} accuracy: {accuracy[i]:.3f}, error: {errors[i]:.3f}",end="\r")
        if self.ema:
            return errors,accuracy,average,means
        else:
            return errors,accuracy
def d_abs(x):
    '''the derivative of the ReLu function
    Args:
        x (ndarray): the h matrix
    output:
        m (ndarray): the derivative of h, f'(h)
    '''    
    m = x.copy()
    m[m<=0] = 0
    m[m>0] =1
    return m

def softmax(x):
    '''the softmax function adjusts the inputs to form a probability distribution
       such that it sums to 1
    Args:
        x (ndarray): the h matrix
    output:
        e (ndarray): the f(h) matrix
    ''' 
    e = np.exp(x-np.max(x)) #to prevent overflow
    return e/np.sum(e,axis=0)


##EXAMPLE EXPERIMENT

def single_layer(training_set,comparison_set):
    ''' shows that the average weight update of a single layer perceptron converges to zero '''

    epochs = 100
    eta = 0.05
    batch_size = 50
    slp = MLP(img_size,[],n_labels,epochs,emnist,softmax=False,ema=True)
    errors,accuracy,average,means = slp.train(training_set[0],training_set[1],epochs,batch_size,eta,comparison_set,string="",False,tau=0.01)
    plt.plot(means)
    plt.show()



if __name__ == "__main__":

    emnist = loadmat('emnist-letters-1k.mat')
    x_train = emnist['train_images']
    train_labels = emnist['train_labels']
    ##normalize the training data
    normalized_xtrain = x_train/255
    x_test = emnist['test_images']
    normalized_xtest = x_test/255
    test_labels = emnist['test_labels']

    print(x_train.shape,x_test.shape) #train: 26000 samples - 1000 per letter, 28*28 features each
                                    #test:  6500 samples  - 250  per letter, 28*28 features each
    n_samples, img_size = x_train.shape
    n_labels = 26


    #randomly select 5200 (20%) indices from the training data
    validation_index = np.random.choice(x_train.shape[0],5200, replace=False)
    print(validation_index.shape,x_train.shape)

    #create a validation set from those indicies
    x_validation = normalized_xtrain[validation_index]
    x_val_labels = train_labels[validation_index]
    print(x_validation.shape,x_val_labels.shape)

    #delete the values at those indicies in the training data and labels
    normalized_xtrain = np.delete(normalized_xtrain,validation_index,0)
    train_labels = np.delete(train_labels,validation_index,0)
    print(normalized_xtrain.shape, train_labels.shape,0)

    #one hot encoding
    y_train = np.zeros((train_labels.shape[0], n_labels))
    y_val =  np.zeros((x_val_labels.shape[0], n_labels))
    y_test  = np.zeros((test_labels.shape[0], n_labels))

    for i in range(0,train_labels.shape[0]):   
        y_train[i, train_labels[i].astype(int)]=1

    for i in range(0,test_labels.shape[0]):    
        y_test[i, test_labels[i].astype(int)]=1

    for i in range(0,x_val_labels.shape[0]):    
        y_val[i, x_val_labels[i].astype(int)]=1

    print(y_train.shape,y_val.shape, y_test.shape)


    compare = [x_validation,y_val]
    test_s = [normalized_xtest,y_test]

    ##experiments -- uncommented as will take ages to run

    single_layer([normalized_xtrain,y_train],compare)

