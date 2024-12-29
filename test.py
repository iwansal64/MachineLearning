from random import choice
from neural_network import MachineLearning, ActivationFunctions
from datasets_manager import load_dataset

datasets = load_dataset()
model = MachineLearning(datasets = datasets, num_of_hidden_layer = 2, num_of_nodes_for_hidden_layer = 16)
while True:
    number = input("Give number: ")
    try:
        chosen_input = choice(datasets[number])
        print(model.predict(chosen_input))
    except KeyError:
        exit(0)