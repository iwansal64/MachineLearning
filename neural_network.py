import math, time, json, random
from typing import Dict, List, Callable, Tuple
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
    def __init__(self, starting_activation: float = 0, starting_bias: float = 0, layer_index: int | None = None, neuron_index: int | None = None):
        '''Class for a particular Neuron or Node'''
        self.activation = starting_activation
        self.bias = starting_bias
        self.layer_index = layer_index
        self.neuron_index = neuron_index
        
    def __float__(self) -> float:
        return float(self.activation)
    
    def __repr__(self) -> str:
        return f"Neuron({str(self.activation)})"
    
    def __lt__(self, other):
        return self.activation < other.activation
        
    def __gt__(self, other):
        return self.activation > other.activation

    def __eq__(self, other):
        return self.activation == other.activation
        
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
        
        self.layers: List[List[Neuron]] = [[Neuron( starting_activation = (random.random() * random.choice([-1, 1])), starting_bias = (random.random() * random.choice([-1, 1])), layer_index = layer_index, neuron_index = neuron_index ) for neuron_index in range( input_size if layer_index == 0 else ( output_size if layer_index == num_of_hidden_layer + 1 else num_of_nodes_for_hidden_layer ))] for layer_index in range(num_of_hidden_layer + 2)]
        '''
        `self.layers[layer index][neuron index]`
        '''
        
        self.weights: List[List[List[Weight]]] = [[[Weight(neuron = neuron, previous_layer_neuron = previous_layer_neuron, starting_value = (random.random() * random.choice([-1, 1]))) for previous_layer_neuron in self.layers[layer_index - 1]] if layer_index > 0 else [0] for neuron in self.layers[layer_index]] for layer_index in range(len(self.layers))]
        '''
        `self.weights[layer index][neuron index][neuron index in the previous layer that connects to neuron in second index]`
        '''

        '''Performing Forward Propagation to get the output of current model for neural network'''
    def forward_propagation(self, inputs: List[float], learn: bool = False) -> List[Neuron] | List[List[Neuron]]:

        if len(inputs) != len(self.layers[0]):
            raise Exception(f"the current input layer length : {len(self.layers[0])} is not as same as the input length given : {len(inputs)}")

        self.layers[0] = [Neuron(starting_activation = value, layer_index = 0, neuron_index = index) for index, value in enumerate(inputs)]

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
                self.layers[current_layer_index][neuron_index].activation = result
                


        if not learn:
            return self.layers[-1]
        else:
            return self.layers
                

class MachineLearning(NeuralNetwork):
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
        
        #? Creating Neural Network
        super().__init__(
            input_size = len(datasets[list(datasets.keys())[0]][0])*len(datasets[list(datasets.keys())[0]][0][0]), 
            output_size = len(datasets.keys()), 
            num_of_hidden_layer = num_of_hidden_layer, 
            num_of_nodes_for_hidden_layer = num_of_nodes_for_hidden_layer,
            activation_function = activation_function
        )
        
        
        self.datasets = datasets
        self.activation_function = activation_function
        self.activation_data = [[Neuron(starting_activation = 0, layer_index = layer_index, neuron_index = neuron_index) for neuron_index in range(len(self.layers[layer_index]))] for layer_index in range(len(self.layers))]
        '''
        `self.activation_data[layer index][neuron index]`
        '''

        self.biases_evaluations = [[float(0) for neuron in self.layers[layer_index]] for layer_index in range(len(self.layers))]
        '''
        `self.bias_evaluations[layer index][neuron index]`
        '''
        
        self.weights_evaluations = [[[float(0) for second_neuron in self.weights[layer_index][first_neuron]] for first_neuron in range(len(self.weights[layer_index]))] for layer_index in range(len(self.layers))]
        '''
        `self.weights_evaluations[layer index][neuron index][neuron index in the previous layer that connects to neuron in second index]`
        '''
        
    def predict(self, input_data: List[float]|List[List[float]]) -> str:
        '''Predict the output of input'''
        try:
            if input_data[0][0]:
                pass
                
            input_data = NeuralNetworkUtils.process_raw_input(input_data)
        except TypeError:
            pass

            
        result = self.forward_propagation(input_data)
        #? --- COMMENT FOR DEBUGGING ---
        result_index = result.index(max(result))
        result = list(self.datasets.keys())[result_index]
        #? ------------------------------
        return result

    def self_evaluation(self, neuron: Neuron, previous_cost: float) -> float:
        '''Evaluate a particular neuron to improve it and the following parameters behind it\n
        Output:
        Two values inside a tuple:
        1. A list of index that directly influence current neuron in previous layer
        2. The activation influence of the neuron you gave in respect of previous_cost\n
        ===== PLAN =====
        ```
        1.Evaluate previous activation influence
        2.Evaluate bias influence
        3.Evaluate weight influence
        4.Update feedback
        5.Evaluate activation influence (recursive)
        ```
        '''
        
        #? 1. Evaluate previous activation influence
        previous_activation_influence = previous_cost * self.activation_function.inverse(neuron.activation)
        
        #? 2. Evaluate bias influence
        bias_influence = previous_activation_influence
        
        #? 3. Evaluate weight influence for current neuron
        weight_influence: List[float] = []
        for weight_index in range(len(self.weights[neuron.layer_index][neuron.neuron_index])):
            weight_influence.append(previous_activation_influence * float(self.activation_data[neuron.layer_index - 1][weight_index]))

        #? 4. Update Feedback
        self.biases_evaluations[neuron.layer_index][neuron.neuron_index] += bias_influence
        self.biases_evaluations[neuron.layer_index][neuron.neuron_index] /= 2

        for weight_index in range(len(self.weights[neuron.layer_index][neuron.neuron_index])):
            self.weights_evaluations[neuron.layer_index][neuron.neuron_index][weight_index] += weight_influence[weight_index]
            self.weights_evaluations[neuron.layer_index][neuron.neuron_index][weight_index] /= 2

        #? 5. Evaluate Activation Influence (recursion happens here :D)
        return previous_activation_influence

    
    def back_propagation(self, results: List[float], correct: List[float]):
        '''Back Propagation (Warning! Calculus ahead! 🤒🤧)
        ALRIGHT HERE'S THE THING:
        1. Get the error :D
        2. Iterate each output neuron
        3. And compute the gradient of that neuron descending (weight, biases, and activation neuron behind it)
        \n\n
        Weight Gradient Formula for neuron-i in layer n-2 to neuron-j in layer n-1:
           dE        dE       dZn-1       dAn-1
        -------- = ------- = ------- = ----------
        dwij(n-1)   dZn-1     dAn-1     dwij(n-1)

        (I KNOW IT'S COMPLICATED RIGHT?)

        Bias Gradient Formula for neuron-i in layer n-2:
           dE       dE       dZn-1     dAn-1
        ------- = ------- = ------- = -------
        dB(n-1)    dZn-1     dAn-1    dB(n-1)
        '''
        errors = [2*(results[i] - correct[i]) for i in range(len(results))]
        
        activation_costs: List[List[float]] = [[([], 0) for neuron_index in range(len(self.layers[layer_index]))] for layer_index in range(len(self.layers))]
        '''
        Used for storing costs
        `activation_costs[layer_index][neuron_index][0 -> List[neuron_index], 1 -> activation_cost (used to recursion)]`
        '''

        for output_index in range(len(results)): #? Iterate each output neuron's index
            output_neuron = self.layers[-1][output_index] #? Get the current output neuron
            corresponding_error = errors[output_index] #? Get the corresponding error for current output neuron
            activation_costs[-1][output_index] = self.self_evaluation(neuron = output_neuron, previous_cost = corresponding_error) #? Evaluate current output neuron (a big weights and biases changes going on here)

        for layer_index in [i for i in range(len(self.layers)-2, 1, -1)]: #? FOR LOOP #1: Iterate each hidden layer from the last hidden layer to the first hidden layer to get the neurons inside it and re evaluate it all
            for neuron_index in range(len(self.layers[layer_index])): #? FOR LOOP #2: Iterate each neuron inside current layer that then wants to be evaluated!
                for second_neuron_index in range(len(self.layers[layer_index+1])): #? FOR LOOP #3: Iterate each neuron inside current layer in the layer forward that used to be the parameters for current neuron index
                    # print(f"{neuron_index}, {second_neuron_index}")
                    activation_costs[layer_index][neuron_index] = self.self_evaluation(neuron = self.layers[layer_index][neuron_index], previous_cost = activation_costs[layer_index + 1][second_neuron_index]) #? Evaluate current neuron

            
        # print(self.biases_evaluations)
        # print(self.weights_evaluations)
    
    def learn(self):
        '''Try to LEARN the datasets! :D'''
        for answer, dataset in self.datasets.items():
            for data in dataset:
                data = NeuralNetworkUtils.process_raw_input(input_data = data)
                self.activation_data: List[List[Neuron]] = self.forward_propagation(data, True)
                self.back_propagation(results = [neuron.activation for neuron in self.activation_data[-1]], correct = [1 if key == answer else 0 for key in list(self.datasets.keys())])

        
if __name__ == "__main__":
    current_datasets = {"1":[],"2":[],"3":[],"4":[],"5":[],"6":[],"7":[],"8":[],"9":[],"0":[]}
    with open(DATASET_FILE_PATH, "r+") as f:
        current_datasets = json.loads(f.read())
    
    neural_network = MachineLearning(datasets = current_datasets, activation_function = ActivationFunctions.Sigmoid())
    neural_network.predict(current_datasets["1"][0])