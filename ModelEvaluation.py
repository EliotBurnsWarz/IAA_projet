import torch
import numpy as np
import matplotlib.pyplot as plt

def eval_model(model, eval_dataset, n_most_likely_cats = 1, type_of_dataset="validation"):
    model.eval()
    correct = 0
    total = 0
    list_of_cat_ranks = []
    with torch.no_grad():
        for images, labels in eval_dataset:
            images = images.unsqueeze(0)
            outputs = model(images)
            total += 1
            number_label = np.argmax(labels).item()
            if (n_most_likely_cats == 1):
                _, predicted = torch.max(outputs.data, 1)
                if number_label == predicted:
                    correct += 1
            else:
                predicted = np.flip(np.argsort(outputs.data)[0][-n_most_likely_cats:].numpy())
                if number_label in predicted:
                    correct += 1
                    indices = np.where(predicted == number_label)
                    list_of_cat_ranks.append(indices[0][0])

    accuracy = 100 * correct / total
    if n_most_likely_cats == 1:
        print(f'Accuracy on {type_of_dataset} dataset: {accuracy:.2f}%')
    else:
        print(f'Likelyhood of finding correct cat in the most likely {n_most_likely_cats} cats predicted by the model, on {type_of_dataset} dataset: {accuracy:.2f}%')
        bin_edges = np.linspace(- 0.5, n_most_likely_cats + 0.5, n_most_likely_cats + 1)
        plt.hist(list_of_cat_ranks, bins=bin_edges, align='mid', alpha=0.75, color='blue', edgecolor='black')

        plt.xlabel('Rang de prédiction')
        plt.ylabel('Fréquence')
        plt.title('Fréquence du rang de prédiction')
        plt.show()

    model.train()
    return accuracy

