import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import RocCurveDisplay, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

def split_df(df, test_size):
#split df into valid and test set
    val_size=int(0.5*len(df))

    #Randomly split indices for training and testing sets
    indices = list(range(len(df)))
    random.shuffle(indices)
    val_indices, test_indices = indices[:val_size], indices[val_size:]

    #Create subset datasets and data loaders for training and testing
    val_dataset = df.iloc[val_indices]
    test_dataset = df.iloc[test_indices]

    return val_dataset, test_dataset


def find_threshold(df, classes_cutoff, min_class, max_class):
    #take columns nr 8 to 10 and sum probabilities
    min_number_of_column=min_class+1
    max_number_of_column=max_class+1
    df.loc[:, 'probability']=df.iloc[:, min_number_of_column:max_number_of_column+1].sum(axis=1)
    
    FP=0
    threshold=0
    #find the threshold- minimum value of probability when we don't have any false positives
    for i in np.arange(0.01,0.98,0.01):
        df.loc[:,'predicted']=df['probability']>i
        #FP -> when true label is 0 and predicted is 1
        FP=len(df[(df['true_label']<=classes_cutoff) & (df['predicted']==1)])
        
        if FP==0 and int(len(df[df['true_label']<=classes_cutoff])>0):
            threshold=i
            break
    '''
    if threshold==0:
        for i in np.arange(0.01,0.98,0.01):
            df.loc[:,'predicted']=df['probability']>i
            #FP -> when true label is 0 and predicted is 1
            FP=len(df[(df['true_label']<=classes_cutoff) & (df['predicted']==1)])
            #print(FP)
            if FP==1 and int(len(df[df['true_label']<=classes_cutoff])>0):
                threshold=i
                break
    '''
    if threshold==0:
        threshold=None
    
    df.loc[:,'predicted']=df['probability']>threshold
    TP=len(df[(df['probability']>threshold) & (df['predicted']==1)])
    print(TP)
    
    gain=TP/len(df['true_label'])
    print(gain)
    return threshold



def check_on_test_set(df, threshold, classes_cutoff,  min_class, max_class):

    if threshold==None:
        return None, None, None, None, None
    
    min_number_of_column=min_class+1
    max_number_of_column=max_class+1
    df.loc[:, 'probability']=df.iloc[:, min_number_of_column:max_number_of_column+1].sum(axis=1)
    
    df.loc[:,'predicted']=df['probability']>threshold
    FP=len(df[(df['true_label']<=classes_cutoff) & (df['predicted']==1)])
    #zmiana z tym że tak naprawdę ok są dla nas te też z tym 1 procenctem niższe
    FP_mod=len(df[(df['true_label']<=(classes_cutoff-1)) & (df['predicted']==1)])
    print("ilosc takich ktore sa rowne lub ponizej progu klasy- prawdziwe 0", len(df[df['true_label']<=classes_cutoff]))
    print("ilosc takich ktore sa powyzej progu klasy- prawdziwe 1", len(df[df['true_label']>classes_cutoff]))
    print("ile przwidziano jako 1", len(df[df['predicted']==1]))
    print("FP check", FP)
    TP=len(df[(df['true_label']>classes_cutoff) & (df['predicted']==1)])
    print("TP check", TP)
    gain=TP/len(df['true_label'])
    TN=len(df[(df['true_label']<=classes_cutoff) & (df['predicted']==0)])
    if TN+FP==0:
        return None, None, None, None,None
    
    specificity=TN/(FP+TN)
    specificity_mod=TN/(FP_mod+TN)

    FN=len(df[(df['true_label']>classes_cutoff) & (df['predicted']==0)])
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)

    return FP, gain, specificity, f1, specificity_mod


def calculate_threshold_and_find_gain(df, val_size, classes_cutoffs, min_class, max_class, nr_of_splits):
    thresholds=[]
    gains=np.zeros((nr_of_splits,len(classes_cutoffs)))
    FPs=np.zeros((nr_of_splits,len(classes_cutoffs)))
    specificities=np.zeros((nr_of_splits,len(classes_cutoffs)))
    f1s=np.zeros((nr_of_splits,len(classes_cutoffs)))
    specificities_mod=np.zeros((nr_of_splits,len(classes_cutoffs)))
                           
    for i in range(nr_of_splits):
        valid_set, test_set=split_df(df, val_size)
        print("i", i)
        for j in range (len(classes_cutoffs)):
            print("j",j)
            threshold=find_threshold(valid_set, classes_cutoff= classes_cutoffs[j], min_class=min_class, max_class=max_class)
            thresholds.append(threshold)
            
            FP,gain,specificity, f1, specificity_mod =check_on_test_set(test_set, threshold, classes_cutoffs[j], min_class, max_class)

            #dla różnych podziałów i 5 podziałów
            if FP is not None and gain is not None:
                gains[i, j] = gain
                FPs[i, j] = FP
                specificities[i, j]=specificity
                f1s[i, j]=f1
                specificities_mod[i, j]=specificity_mod

            else:
                gains[i, j] = np.nan
                FPs[i, j] = np.nan
                specificities[i, j]=np.nan
                f1s[i, j]=np.nan
                specificities_mod[i, j]=np.nan


    return gains, FPs, specificities, f1s, specificities_mod

    gain_final=np.nanmean(gains, axis=0)
    gain_std=np.nanstd(gains, axis=0)
    FP_final=np.nanmean(FPs, axis=0)
    FP_std=np.nanstd(FPs, axis=0)
    specificity=np.nanmean(specificities, axis=0)
    specificity_std=np.nanstd(specificities, axis=0)

    #gains_final.append(gain_final)
    #FPs_final.append(FP_final)
    #print(gains_final)
    #print(FPs_final)

    #dla każdego podzbioru
    '''
    #plot gains and FPs in the function od classes_cutoff
    plt.figure(figsize=(10,5))
    plt.errorbar(classes_cutoffs, gain_final, yerr=gain_std, fmt='o')
    plt.xlabel('Classes cutoff')
    plt.ylabel('Gain')
    plt.title('Gain for different classes cutoff')
    plt.show()

    plt.figure(figsize=(10,5))
    plt.errorbar(classes_cutoffs, FP_final, yerr=FP_std, fmt='o')
    plt.xlabel('Classes cutoff')
    plt.ylabel('FP')
    plt.title('FP for different classes cutoff')
    plt.show()
    '''
    return gain_final, gain_std, FP_final, FP_std, specificity, specificity_std


def calculate_average_gain_and_FP(csv_file, val_size, classes_cutoffs, min_class, max_class, nr_of_splits):
    df=pd.read_csv(csv_file, sep=',')
    gains, FPs, specificities,f1s, specificities_mod=calculate_threshold_and_find_gain(df, val_size, classes_cutoffs, min_class, max_class, nr_of_splits)
    return gains, FPs, specificities,f1s,specificities_mod


#od tad juz nie
def calculate_number_of_data_in_classes(root):
    classes=[0,1,2,3,4,5,6,7,8,9]
    class_counts = {}
    count=0
    for class_name in classes:
        
        path=os.path.join(root, str(class_name))
        count=len(os.listdir(path))
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += count

    return class_counts    

def calculate_normal_scores(df, classes_cutoff):
    #df.loc[:, 'predicted'] = df.iloc[:, 1:11].idxmax(axis=1).astype(int)
    df_values=df.iloc[:, 1:11]
    df['predicted'] =df_values.idxmax(axis=1).astype(int)
  
    #df['true_0_1']=df['true_label']>classes_cutoff

    FPs=np.zeros(len(classes_cutoff))
    gains=np.zeros(len(classes_cutoff))
    specificities=np.zeros(len(classes_cutoff))
    f1s=np.zeros(len(classes_cutoff))
    roc_curves = []


    # Assuming 'pred_value' and 'df_values' are correctly defined previously

    for i, class_cutoff in enumerate(classes_cutoff):
        for index, row in df.iterrows():
            pred_value = row['predicted']
            if pred_value > class_cutoff:
                df.loc[index, 'probability'] = df_values.loc[index, str(class_cutoff):].sum()
            else:
                df.loc[index, 'probability'] = df_values.loc[index, :str(class_cutoff)].sum()


        TP=len(df[(df['predicted']>class_cutoff) & (df['true_label']>class_cutoff)])
        FP=len(df[(df['predicted']>class_cutoff) & (df['true_label']<=class_cutoff)])
        TN=len(df[(df['predicted']<=class_cutoff) & (df['true_label']<=class_cutoff)])
        FN=len(df[(df['predicted']>class_cutoff) & (df['true_label']<=class_cutoff)])
        gain=TP/len(df['true_label'])
        specificity=TN/(FP+TN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        FPs[i]=FP
        gains[i]=gain
        specificities[i]=specificity
        f1s[i]=f1

        fpr, tpr, _ = roc_curve(df['true_label'] > class_cutoff, df['predicted']> class_cutoff)
        print(len(fpr))
        roc_curves.append((fpr, tpr, auc(fpr, tpr)))

        print("TP", TP)
        print("FP", FP)
        print("TN", TN)
        print("FN", FN)
        print("gain", gains)
        print("specificity", specificities)
        print("f1", f1s)

    return gains, FPs, specificities, f1s, roc_curves


def calculate_normal_scores_mod(df, classes_cutoff, min_class, max_class):

    min_number_of_column=min_class+1
    max_number_of_column=max_class+1
    df.loc[:, 'probability']=df.iloc[:, min_number_of_column:max_number_of_column+1].sum(axis=1)

    FPs=np.zeros(len(classes_cutoff))
    gains=np.zeros(len(classes_cutoff))
    specificities=np.zeros(len(classes_cutoff))
    f1s=np.zeros(len(classes_cutoff))
    for i, class_cutoff in enumerate(classes_cutoff):

        TP=len(df[(df['predicted']>class_cutoff) & (df['true_label']>class_cutoff)])
        FP=len(df[(df['predicted']>class_cutoff) & (df['true_label']<=class_cutoff)])
        TN=len(df[(df['predicted']<=class_cutoff) & (df['true_label']<=class_cutoff)])
        FN=len(df[(df['predicted']>class_cutoff) & (df['true_label']<=class_cutoff)])
        gain=TP/len(df['true_label'])
        specificity=TN/(FP+TN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        FPs[i]=FP
        gains[i]=gain
        specificities[i]=specificity
        f1s[i]=f1
        print("TP", TP)
        print("FP", FP)
        print("TN", TN)
        print("FN", FN)
        print("gain", gains)
        print("specificity", specificities)
        print("f1", f1s)

    return gains, FPs, specificities, f1s


def find_threshold_class(df, classes_cutoff, min_class, max_class):
    df.loc[:, 'predicted'] = df.iloc[:, 1:11].idxmax(axis=1).astype(int)
    print(df['predicted'])
    print(len(df[df['predicted']==4]))
    FP=0
    threshold=0
    classes_cutoff=sorted(classes_cutoff, reverse=True)
    print("dp",classes_cutoff)
    #find the threshold- minimum value of probability when we dont have any false positives
    #chcemy jak najmniejszą liczbę, ale trzeba iść od góry
    for class_cutoff in classes_cutoff:
        #FP -> when true label is 0 and predicted is 1
        print(class_cutoff)
        FP=len(df[(df['true_label']<=class_cutoff) & (df['predicted']>class_cutoff)])
        print("wartosc fp", FP)
        #FP=len(df[(df['true_label']<=classes_cutoff) & (df['predicted']==1)])
        #print(FP)
        if FP==0:
            threshold=class_cutoff
            break
    print("thr",threshold)
    return threshold



def check_on_test_set_class(df, threshold, classes_cutoff,  min_class, max_class):
    if threshold==0:
        print("nie znaleziono takiego progu")
        return 100,100
    df.loc[:, 'predicted'] = df.iloc[:, 1:11].idxmax(axis=1).astype(int)>threshold

    FP=len(df[(df['true_label']<=threshold) & (df['predicted']>threshold)])
    TP=len(df[(df['true_label']>threshold) & (df['predicted']>threshold)])
    gain=TP/len(df['true_label'])

    return FP, gain


def calculate_threshold_and_find_gain_class(df, val_size, classes_cutoffs, min_class, max_class):
    gains=np.zeros(5)
    print("shape",gains.shape)
    FPs=np.zeros(5)

    gains_final=[]
    FPs_final=[]
    gains_std=[]
    FPs_std=[]
    thresholds=[]
    for i in range(5):
        valid_set, test_set=split_df(df, val_size)
        print("i", i)
        threshold=find_threshold_class(valid_set, classes_cutoff= classes_cutoffs, min_class=min_class, max_class=max_class)
        thresholds.append(threshold)
        FP,gain=check_on_test_set_class(test_set, threshold, classes_cutoffs, min_class, max_class)
        #dla różnych podziałów i 5 podziałów
        gains[i]=gain
        FPs[i]=FP
    print('thresholds',thresholds) 
    print('gains',gains)
    print('fps',FPs)
    gain_final=np.mean(gains, axis=0)
    gain_std=np.std(gains, axis=0)
    FP_final=np.mean(FPs, axis=0)
    FP_std=np.std(FPs, axis=0)

    #gains_final.append(gain_final)
    #FPs_final.append(FP_final)
    #print(gains_final)
    #print(FPs_final)
