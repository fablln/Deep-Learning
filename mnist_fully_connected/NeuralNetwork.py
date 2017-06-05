import time
import random
import numpy as np
from utils import *
from transfer_functions import * 


class NeuralNetwork(object):
    
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, iterations=50, learning_rate = 0.1,transfer_function = sigmoid,d_transfer_function = dsigmoid):
        """
        input: number of input neurons
        hidden: number of hidden neurons
        output: number of output neurons
        iterations: how many iterations
        learning_rate: initial learning rate
        """
       
        # initialize parameters
        self.iterations = iterations   #iterations
        self.learning_rate = learning_rate
        self.tf = transfer_function
        self.dtf = d_transfer_function
    
    
        # initialize arrays sizes
        self.input = input_layer_size+1  # +1 for the bias node in the input Layer
        self.hidden = hidden_layer_size+1 #+1 for the bias node in the hidden layer 
        self.output = output_layer_size

        # set up array of 1s for activations
        self.a_input = np.ones(self.input)
        self.a_hidden = np.ones(self.hidden)
        self.a_out = np.ones(self.output)
        
        #create randomized weights Yann Lecun method in 1988's paper ( Default values)
        input_range = 1.0 / self.input ** (1/2)
        self.W_input_to_hidden = np.random.normal(loc = 0, scale = input_range, size =(self.input, self.hidden-1))
        self.W_hidden_to_output = np.random.uniform(size = (self.hidden, self.output)) / np.sqrt(self.hidden)
 

    def set_transfer_function(self, transfer_function, d_transfer_function):
        self.tf = transfer_function
        self.dtf = d_transfer_function
        
    def weights_initialisation(self,wi,wo):
        self.W_input_to_hidden=wi # weights between input and hidden layers
        self.W_hidden_to_output=wo # weights between hidden and output layers
        


    #========================Begin implementation section 1============================================="    
    
    def feedForward(self, inputs):
        # Compute input activations
        self.a_input = np.append(inputs, [1])
        # Compute  hidden activations
        self.a_hidden = np.append(self.tf(self.a_input.dot(self.W_input_to_hidden)), [1])
        # Compute output activations       
        self.a_out = self.tf(self.a_hidden.dot(self.W_hidden_to_output))
        
        return self.a_out
     #========================End implementation section 1==============================================="   
        
        
        
     #========================Begin implementation section 2=============================================#    

    def backPropagate(self, targets):

        # calculate error terms for output
        self.err_out = self.a_out - targets
        # calculate error terms for hidden
        delta_out = self.err_out * self.dtf(self.a_out)
        delta_hidden = self.W_hidden_to_output.dot(delta_out) * self.dtf(self.a_hidden)
        # update output weights: calculate the new weights
        self.W_hidden_to_output -= self.learning_rate * np.outer(self.a_hidden, delta_out)
        # update input weights
        self.W_input_to_hidden -= self.learning_rate * np.outer(self.a_input, delta_hidden[:-1])
        # calculate error
        return np.sum(self.err_out**2) / 2

    
    #========================End implementation section 2 =================================================="   

    
    def train(self, data,validation_data):
        start_time = time.time()
        errors=[]
        Training_accuracies=[]
        Validation_accuracies=[]
      
        for it in range(self.iterations):
            np.random.shuffle(data)
            inputs  = [entry[0] for entry in data ]
            targets = [ entry[1] for entry in data ]
            
            error=0.0 
            for i in range(len(inputs)):
                Input = inputs[i]
                Target = targets[i]
                self.feedForward(Input)
                error+=self.backPropagate(Target)
            
            Training_accuracies.append(self.predict(data)/len(data)*100)
            Validation_accuracies.append(self.predict(validation_data)/len(validation_data)*100)
            
            error=error/len(data)
            errors.append(error)
            
           
            print("Iteration: %2d/%2d[==============] -Error: %5.10f  -Training_Accuracy:  %2.2f  -time: %2.2f " %(it+1,self.iterations, error, (self.predict(data)/len(data))*100, time.time() - start_time))
            # you can add test_accuracy and validation accuracy for visualisation 
        return (errors,Training_accuracies,Validation_accuracies)
        #plot_curve(range(1,self.iterations+1),errors, "Error")
        #plot_curve(range(1,self.iterations+1), Training_accuracies, "Training_Accuracy")
        #plot_curve(range(1,self.iterations+1), Validation_accuracies, "Validation_Accuracy")
       

    def predict(self, test_data):
        """ Evaluate performance by counting how many examples in test_data are correctly 
            evaluated. """
        count = 0.0
        for testcase in test_data:
            answer = np.argmax( testcase[1] )
            prediction = np.argmax( self.feedForward( testcase[0] ) )
            count = count + 1 if (answer - prediction) == 0 else count 
            count= count 
        return count 
    
    def predict2(self, test_data):
        """ Evaluate performance by counting how many examples in test_data are correctly 
            evaluated. """
        answer = test_data[1]
        prediction = np.argmax(self.feedForward(test_data[0]))
        return (prediction,answer)
    
    
    def save(self, filename):
        """ Save neural network (weights) to a file. """
        with open(filename, 'wb') as f:
            pickle.dump({'wi':self.W_input_to_hidden, 'wo':self.W_hidden_to_output}, f )
        
        
    def load(self, filename):
        """ Load neural network (weights) from a file. """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        # Set biases and weights
        self.W_input_to_hidden=data['wi']
        self.W_hidden_to_output = data['wo']