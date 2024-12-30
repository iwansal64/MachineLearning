from random import choice
from neural_network import MachineLearning, ActivationFunctions, NeuralNetworkUtils
from datasets_manager import load_dataset
from typing import List

datasets = load_dataset()
model = MachineLearning(datasets = datasets, num_of_hidden_layer = 2, num_of_nodes_for_hidden_layer = 16)
model.learn()

def automatic_test_datasets(machine_learning_model: MachineLearning):
    processed_datasets = NeuralNetworkUtils.process_raw_datasets(datasets)
    right_answers: List[str] = []
    total_questions: int = 0
    for answer in processed_datasets.keys():
        for test in processed_datasets[answer]:
            result = model.predict(test)
            if answer == result:
                right_answers.append(result)
            total_questions += 1
                
    print(f"Result: {len(right_answers)}/{total_questions}")
    print(f"Percentage: {len(right_answers)/total_questions*100}%")

def manual_test_datasets():
    while True:
        number = input("Give number: ")
        try:
            chosen_input = choice(datasets[number])
            print(model.predict(chosen_input))
        except KeyError:
            exit(0)

automatic_test_datasets(model)