import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

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
    #print(df['probability'])
    FP=0
    threshold=0
    #find the threshold- minimum value of probability when we dont have any false positives
    for i in np.arange(0.01,0.98,0.01):
        df.loc[:,'predicted']=df['probability']>i
        #FP -> when true label is 0 and predicted is 1
        FP=len(df[(df['true_label']<=classes_cutoff) & (df['predicted']==1)])
        #print(FP)
        if FP==0 and int(len(df[df['true_label']<=classes_cutoff])>0):
            threshold=i
            break
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
        return None, None
    min_number_of_column=min_class+1
    max_number_of_column=max_class+1
    df.loc[:, 'probability']=df.iloc[:, min_number_of_column:max_number_of_column+1].sum(axis=1)
    df.loc[:,'predicted']=df['probability']>threshold
    FP=len(df[(df['true_label']<=classes_cutoff) & (df['predicted']==1)])
    print("ilosc takich ktore sa rowne lub ponizew progu klasy", len(df[df['true_label']<=classes_cutoff]))
    print("ile przwidziano jako 1", len(df[df['predicted']==1]))
    print("FP check", FP)
    TP=len(df[(df['true_label']>classes_cutoff) & (df['predicted']==1)])
    print("TP check", TP)
    gain=TP/len(df['true_label'])

    return FP, gain


def calculate_threshold_and_find_gain(df, val_size, classes_cutoffs, min_class, max_class, nr_of_splits):
    gains=np.zeros((nr_of_splits,len(classes_cutoffs)))
    print("shape",gains.shape)
    FPs=np.zeros((nr_of_splits,len(classes_cutoffs)))

    gains_final=[]
    FPs_final=[]
    gains_std=[]
    FPs_std=[]
    thresholds=[]
    for i in range(nr_of_splits):
        valid_set, test_set=split_df(df, val_size)
        print("i", i)
        for j in range (len(classes_cutoffs)):
            print("j",j)
            threshold=find_threshold(valid_set, classes_cutoff= classes_cutoffs[j], min_class=min_class, max_class=max_class)
            thresholds.append(threshold)
            
            FP,gain=check_on_test_set(test_set, threshold, classes_cutoffs[j], min_class, max_class)
            #dla różnych podziałów i 5 podziałów
            if FP is not None and gain is not None:
                gains[i, j] = gain
                FPs[i, j] = FP
            else:
                gains[i, j] = np.nan
                FPs[i, j] = np.nan

    print(thresholds)
    print(FPs)
    print(gains)
    gain_final=np.nanmean(gains, axis=0)
    gain_std=np.nanstd(gains, axis=0)
    FP_final=np.nanmean(FPs, axis=0)
    FP_std=np.nanstd(FPs, axis=0)

    #gains_final.append(gain_final)
    #FPs_final.append(FP_final)
    #print(gains_final)
    #print(FPs_final)

    #dla każdego podzbioru

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
