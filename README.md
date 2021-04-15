Multi Layer Perceptron to classify EMNIST

by running the python file (python mlp.py) you will automatically run the first test of the single layer perceptron, as an example. 

To create your own MLP, initalise the object (values used are arbitrary, might not work well with EMNIST):

my_mlp = MLP(50,[100,200],15,50,emnist,softmax=True,ema=False)
this creates an object with input of 50 nodes (number of features for emnist is 28*28)
two hidden layers of 100 and 200 nodes respectively
an output layer of 15 nodes (emnist has 26 classes)
default value for number of epochs but you can override this when running training
softmax=true uses softmax activation function on the output (uses ReLU otherwise)
ema = True will calculate the exponential moving average, this will only work for single layer perceptrons. attempting to use this with a MLP will remove your hidden layers and convert to SLP


In this assignment I used a subsection of the EMNIST data set:

G.Cohen,S.Afshar,J.Tapson,andA.vanSchaik,“EMNIST:an extension of MNIST to hand written letters.”Retrieved from:http://arxiv.org/abs/1702.05373,(2017)


