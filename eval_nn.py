import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import random
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import pandas as pd
import matplotlib.pyplot as plt


def eval(test_loader, model,num_classes):
    model.eval()
    correct = 0
    total = 0
    all_labels=[]
    all_predicted=[]
    multiplier=0.8
    all_probabilities = []
    all_labels_0_1 = []
    all_probabilities_0_1 = []
    TP=0
    TN=0
    FP=0
    FN=0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            
            if num_classes>2:
                probabilities = torch.softmax(outputs, dim=1)  #Przekształć wyniki na prawdopodobieństwa za pomocą funkcji softmax
                #macierz 10x400
                modified_probabilities = probabilities.clone()  #Skopiuj wyniki, aby uniknąć modyfikowania oryginalnych danych
                #modified_probabilities[:, 9] *= multiplier  #Pomnóż prawdopodobieństwo klasy 9 przez współczynnik
                all_probabilities.extend(modified_probabilities.cpu().numpy())
                #from class 7 to 9 we have 1 class-> I sum the probabilities
                probability_1 = modified_probabilities[:, 7:].sum(axis=1)
                all_probabilities_0_1.extend(probability_1)
                #all_predicted_0_1 = all_probabilities_0_1 > 0.5 #klasyfikacja 0 1 na podstawie prawdopodobieństw
                #all_predicted_0_1 = np.array(all_probabilities_0_1) > 0.5
                #jaka musi być wartość prawdopodobienstwa np klasy 9 żeby FP było zawsze 0, jeszcez trzeba uznac od ktorej klasy np od 4
                #jaki jest gain , podzial csv na zbior walidacyjny i testowy
                #all_predicted_0_1=all_predicted_0_1.cpu().numpy()


                _, predicted = torch.max(modified_probabilities, 1)  #Wybierz klasyfikację na podstawie prawdopodobieństw z 10 klas
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            '''
            else:
                thresholded_outputs = (outputs > 0.98).int()
                labels = (labels > 98).int()
                total += labels.size(0)
                correct += (thresholded_outputs == labels).sum().item()
            '''

            #labels
            labels = labels.data.cpu().numpy()
            all_labels.extend(labels) # Save Truth
            all_labels_0_1.extend(labels > 6) # Save 0/1 labels

            #predicted
            if num_classes > 2:
                all_predicted.extend(predicted.cpu().numpy())
            '''    
            else:
                all_predicted.extend(thresholded_outputs.cpu().numpy())
            '''
    all_probabilities = np.array(all_probabilities)
    #all_predicted_0_1=np.array(all_predicted_0_1)
    df=pd.DataFrame(all_probabilities)
    df['true_label']=all_labels
    df['1_probability']=all_probabilities_0_1
    df.to_csv('probabilities_1.csv')
    all_predicted = np.array(all_predicted)

    #confusion matrix
    confusion_mat = confusion_matrix(all_labels, all_predicted, labels=range(num_classes))
    print(confusion_mat)

    #accuracy
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

    #roc curve
    all_labels_0_1 = np.array(all_labels_0_1)
    all_probabilities_0_1 = np.array(all_probabilities_0_1)
    fpr, tpr, thresholds = roc_curve(all_labels_0_1, all_probabilities_0_1)
    roc_auc = roc_auc_score(all_labels_0_1, all_probabilities_0_1)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    #save figure
    plt.savefig('roc.png')

    '''
    #tp tn fp fn
    TP = np.sum((all_labels_0_1 == 1) & (all_predicted_0_1 == 1))
    TN = np.sum((all_labels_0_1 == 0) & (all_predicted_0_1 == 0))
    FP = np.sum((all_labels_0_1 == 0) & (all_predicted_0_1 == 1))
    FN = np.sum((all_labels_0_1 == 1) & (all_predicted_0_1 == 0))

    print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')

    #F1 score
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f'F1 score: {f1:.4f}')
    '''