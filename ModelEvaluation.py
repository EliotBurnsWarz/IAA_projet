import torch
import numpy as np

def eval_model(model, eval_dataset, type_of_dataset="validation"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in eval_dataset:
            images = images.unsqueeze(0)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            for i in range(len(labels)):
                if predicted == np.argmax(labels):
                    correct += 1

    accuracy = 100 * correct / total
    print(f'Accuracy on {type_of_dataset} dataset: {accuracy:.2f}%')

    model.train()
    return accuracy