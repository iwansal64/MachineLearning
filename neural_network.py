import math, time, json, random
from typing import Dict, List, Callable
from variables import DATASET_FILE_PATH
from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    @abstractmethod
    def __init__(self, f: Callable[[float], float], df: Callable[[float], float]):
        '''Activation Functions Class'''
        self._f = f
        self._df = df
        pass
    
    def __call__(self, x: float):
        return self._f(x)

    def inverse(self, x: float):
        return self._df(x)

class ActivationFunctions:
    class Sigmoid(ActivationFunction):
        def __init__(self):
            '''Sigmoid Function'''
            super().__init__(lambda x: 1 / (1 + math.e**(-x)), lambda x: (1 / (1 + math.e**(-x))) * (1 - (1 / (1 + math.e**(-x)))))
            
    class ReLU(ActivationFunction):
        def __init__(self):
            '''ReLU (Rectified Linear Unit) Function'''
            super().__init__(lambda x: x if x > 0 else 0, lambda x: 1 if x >= 0 else 0)

    class Tanh(ActivationFunction):
        def __init__(self):
            '''Tanh (Hyprbolic Tangent) Function'''
            super().__init__(lambda x: (math.e**(x) - math.e**(-x)) / (math.e**(x) + math.e**(-x)), lambda x: 1 - ((math.e**(x) - math.e**(-x)) / (math.e**(x) + math.e**(-x)))**2)


class NeuralNetworkUtils:
    def process_raw_datasets(datasets: Dict[str, List[List[List[float]]]]) -> Dict[str, List[List[float]]]:
        for key in datasets.keys():
            for dataset in range(len(datasets[key])):
                results = []

                for row in range(len(datasets[key][dataset])):
                    for value in datasets[key][dataset][row]:
                        results.append(value)
                        
                datasets[key][dataset] = results

        return datasets
    
    def process_raw_input(input_data: List[List[float]]) -> List[float]:
        results = []

        for row in range(len(input_data)):
            for value in input_data[row]:
                results.append(value)
                
        return results

class Neuron:
    def __init__(self, starting_value: float = 0, starting_bias: float = 0, layer_index: int | None = None, neuron_index: int | None = None):
        '''Class for a particular Neuron or Node'''
        self.value = starting_value
        self.bias = starting_bias
        self.layer_index = layer_index
        self.neuron_index = neuron_index
        
    def __float__(self) -> float:
        return float(self.value)
    
    def __repr__(self) -> str:
        return f"Neuron({str(self.value)})"
    
    def __lt__(self, other):
        return self.value < other.value
        
    def __gt__(self, other):
        return self.value > other.value

    def __eq__(self, other):
        return self.value == other.value
        
class Weight:
    def __init__(self, neuron: Neuron, previous_layer_neuron: Neuron, starting_value: float = 0):
        '''Class for a connection weight between two neuron'''
        self.first_neuron = neuron
        self.second_neuron = previous_layer_neuron
        self.value = starting_value

    def __float__(self) -> float:
        return float(self.value)

    def __repr__(self) -> str:
        return str(self.value)

class NeuralNetwork:
    def __init__(self, input_size: int, output_size: int, num_of_hidden_layer: int, num_of_nodes_for_hidden_layer: int, activation_function: ActivationFunction):
        '''Class for A network of neurons or nodes'''
        self.num_of_hidden_layer = num_of_hidden_layer
        self.num_of_nodes_on_hidden_layer = num_of_nodes_for_hidden_layer
        self.activation_function = activation_function
        
        #? --- A little bit of IMPORTANT legends here~ ---
        #? self.layers[layer index][neuron index]
        #? self.weights[layer index][neuron index][neuron index in the previous layer that connects to neuron in second index]

        self.layers: List[List[Neuron]] = [[Neuron( starting_value = (random.random() * random.choice([-1, 1])), starting_bias = (random.random() * random.choice([-1, 1])), layer_index = layer_index, neuron_index = neuron_index ) for neuron_index in range( input_size if layer_index == 0 else ( output_size if layer_index == num_of_hidden_layer + 1 else num_of_nodes_for_hidden_layer ))] for layer_index in range(num_of_hidden_layer + 2)]
        self.weights: List[List[List[Weight]]] = [[[Weight(neuron = neuron, previous_layer_neuron = previous_layer_neuron, starting_value = (random.random() * random.choice([-1, 1]))) for previous_layer_neuron in self.layers[layer_index - 1]] if layer_index > 0 else [0] for neuron in self.layers[layer_index]] for layer_index in range(len(self.layers))]
        print("LAYERS")
        print(self.layers)
        print("WEIGHTS")
        print(self.weights)

    def forward_propagation(self, inputs: List[float]) -> List[Neuron]:
        '''Performing Forward Propagation to get the output of current model for neural network'''

        if len(inputs) != len(self.layers[0]):
            raise Exception(f"the current input layer length : {len(self.layers[0])} is not as same as the input length given : {len(inputs)}")

        self.layers[0] = [Neuron(starting_value = value, layer_index = 0, neuron_index = index) for index, value in enumerate(inputs)]

        # Iterate through each layer
        for current_layer_index in range(1, len(self.layers)):
            previous_layer_index = current_layer_index - 1

            # Iterate through each neuron inside current layer
            for neuron_index in range(len(self.layers[current_layer_index])):
                sum_result = 0
                neuron = self.layers[current_layer_index][neuron_index] # Get the First Neuron!
                #? ========= Step by Step of doing a Forward Propagation (for a particular neuron) =========
                #? First Step - Get the Weighted Sum!
                for previous_layer_neuron_index in range(len(self.layers[previous_layer_index])): # Iterate each neuron that connected to the first neuron in previous layer.
                    previous_layer_neuron = self.layers[previous_layer_index][previous_layer_neuron_index] # Get the neuron connected to the First Neuron!
                    weight = self.weights[current_layer_index][neuron_index][previous_layer_neuron_index] # Get the Weight of connection between them!
                    calculation = float(previous_layer_neuron) * float(weight) # Calculate the xi * wij !
                    sum_result += calculation # Sum the result to sum_result. Making the weighted sum for the First Neuron!

                #? Second Step - Add bias to the Weighted Sum
                bias = neuron.bias # Get the First Neuron bias!
                sum_result += bias # Add the bias into the weighted sum!

                #? Third Step - And lastly, add the result (Weighted Sum that added by the bias) into an Activation Function!
                result = self.activation_function(sum_result) # Put the sum_result into the Activation Function and get the result!

                #? Update the neuron value with the result and ready for the next neuron!
                self.layers[current_layer_index][neuron_index].value = result
                


        return self.layers[-1]
                

class MachineLearning:
    def __init__(self, datasets: Dict[str, List[List[List[float]]]], num_of_hidden_layer: int = 2, num_of_nodes_for_hidden_layer: int = 16, activation_function: ActivationFunction = ActivationFunctions.ReLU()):
        '''Class for neural network training.\n
        --- Parameters Informations ---\n
        `dataset`: used for datasets for training refrence.
            it formatted like this:
            {
                "{key1}": [
                    [training data 1 for key1],\n
                    [training data 2 for key1],\n
                    [training data 3 for key1],\n
                    ...,\n
                    [training data n for key1]\n
                ],
                "{key2}": [
                    [training data 1 for key2],\n
                    [training data 2 for key2],\n
                    [training data 3 for key2],\n
                    ...,\n
                    [training data n for key2]\n
                ]
            }
        '''
        self.datasets = datasets
        self.activation_function = activation_function
        
        #? Creating Neural Network
        self.neural_network = NeuralNetwork(
            input_size = len(datasets[list(datasets.keys())[0]][0])*len(datasets[list(datasets.keys())[0]][0][0]), 
            output_size = len(datasets.keys()), 
            num_of_hidden_layer = num_of_hidden_layer, 
            num_of_nodes_for_hidden_layer = num_of_nodes_for_hidden_layer,
            activation_function = activation_function
        )
        
    def predict(self, input_data: List[float]|List[List[float]]) -> List[float]:
        '''Predict the output of input'''
        if input_data[0][0]:
            pass
            
        input_data = NeuralNetworkUtils.process_raw_input(input_data)

            
        result = self.neural_network.forward_propagation(input_data)
        #? --- COMMENT FOR DEBUGGING ---
        result_index = result.index(max(result))
        result = list(self.datasets.keys())[result_index]
        #? ------------------------------
        return result
    
        
        
if __name__ == "__main__":
    current_datasets = {"1":[],"2":[],"3":[],"4":[],"5":[],"6":[],"7":[],"8":[],"9":[],"0":[]}
    with open(DATASET_FILE_PATH, "r+") as f:
        current_datasets = json.loads(f.read())
    
    neural_network = MachineLearning(datasets = current_datasets, activation_function = ActivationFunctions.Sigmoid())
    neural_network.predict(current_datasets["1"][0])