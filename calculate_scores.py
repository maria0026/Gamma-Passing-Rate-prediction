import pandas as pd
import random
import valid
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


classes_cutoffs=[3,4,5]
min_class=7
#max_class=min_class
max_class=9
nr_of_splits=20
nr_of_csv=10
val_size=0.5
csv_files=['dataframes/probabilities_1_4.csv','dataframes/probabilities_2_4.csv','dataframes/probabilities_3_4.csv','dataframes/probabilities_4_4.csv','dataframes/probabilities_5_4.csv','dataframes/probabilities_6_4.csv','dataframes/probabilities_7_4.csv', 'dataframes/probabilities_8_4.csv','dataframes/probabilities_9_4.csv', 'dataframes/probabilities_10_4.csv',  ]


gain=np.zeros((nr_of_csv,len(classes_cutoffs)))
gain_std=np.zeros((nr_of_csv,len(classes_cutoffs)))
FP=np.zeros((nr_of_csv,len(classes_cutoffs)))
FP_std=np.zeros((nr_of_csv,len(classes_cutoffs)))
specificity=np.zeros((nr_of_csv,len(classes_cutoffs)))
specificity_std=np.zeros((nr_of_csv,len(classes_cutoffs)))
f1_score=np.zeros((nr_of_csv, len(classes_cutoffs)))
f1_std=np.zeros((nr_of_csv, len(classes_cutoffs)))
roc_curves_all = np.zeros((nr_of_csv, len(classes_cutoffs)), dtype=object)    
  
all_gains=np.zeros((nr_of_csv*nr_of_splits, len(classes_cutoffs)))
all_FPs=np.zeros((nr_of_csv*nr_of_splits, len(classes_cutoffs)))
all_specificities=np.zeros((nr_of_csv*nr_of_splits, len(classes_cutoffs)))
all_f1_scores=np.zeros((nr_of_csv*nr_of_splits, len(classes_cutoffs)))
all_specificities_mod=np.zeros((nr_of_csv*nr_of_splits, len(classes_cutoffs)))


i=0
for csv_file in csv_files:
    df=pd.read_csv(csv_file, sep=',')
    gains_csv, FPs_csv, specificities_csv, f1_csv, roc_curves=valid.calculate_normal_scores(df, classes_cutoffs)
    gain[i]=gains_csv
    FP[i]=FPs_csv
    specificity[i]=specificities_csv
    f1_score[i]=f1_csv
    roc_curves_all[i]=roc_curves
    i+=1
print("gain",gain)
print("fp",FP)
print("specificity",specificity)

gain_mean=np.nanmean(gain, axis=0)
gain_std=np.nanstd(gain, axis=0)
FP_final_mean=np.nanmean(FP, axis=0)
FP_std=np.nanstd(FP, axis=0)
specificity_mean=np.nanmean(specificity, axis=0)
specificity_std=np.nanstd(specificity, axis=0)
f1_scores_mean=np.nanmean(f1_score, axis=0)
f1_scores_std=np.nanstd(f1_score, axis=0)


print("Gain", gain_mean, gain_std)
print("FP", FP_final_mean, FP_std)
print("Specificity", specificity_mean, specificity_std)
print("F1", f1_scores_mean, f1_scores_std)


# Plot ROC curves for each class cutoff
for j, class_cutoff in enumerate(classes_cutoffs):
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for i in range(nr_of_csv):
        fpr, tpr, roc_auc = roc_curves_all[i, j]
        
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    print(aucs)
    plt.figure(figsize=(6, 6))
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Średnia krzywa ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=0.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label=r'$\pm$ 1 odchylenie standardowe')
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title(f'Średnia krzywa ROC dla klasy odcięcia {class_cutoff}', fontsize=15)
    plt.legend(loc='lower right')
    plt.show()



#plots
plt.figure(figsize=(10,5))
plt.errorbar(classes_cutoffs, gain_mean, yerr=gain_std, fmt='o')
plt.xlabel('Odcięcie klasy', fontsize=15)
plt.ylabel('Gain', fontsize=15)
#plt.title('Gain for different classes cutoff')
plt.show()

plt.figure(figsize=(10,5))
plt.errorbar(classes_cutoffs, FP_final_mean, yerr=FP_std, fmt='o')
plt.xlabel('Odcięcie klasy', fontsize=15)
plt.ylabel('FP', fontsize=15)
#plt.title('FP for different classes cutoff')
plt.show()

plt.figure(figsize=(10,5))
plt.errorbar(classes_cutoffs, specificity_mean, yerr=specificity_std, fmt='o')
plt.xlabel('Odcięcie klasy', fontsize=15)
plt.ylabel('Specyficzność', fontsize=15)
#plt.title('Specificity for different classes cutoff')
plt.show()

plt.figure(figsize=(10,5))
plt.errorbar(classes_cutoffs, f1_scores_mean, yerr=f1_scores_std, fmt='o')
plt.xlabel('Odcięcie klasy', fontsize=15)
plt.ylabel('F1', fontsize=15)
#plt.title('F1 for different classes cutoff')
plt.show()

#histogramy
max_gain=np.nanmax(gain)
max_FP=np.nanmax(FP)
max_spec=np.nanmax(specificity)
max_f1=np.nanmax(f1_score)

max_gain_y = 0
max_FPs_y=0
max_spec_y=0
max_f1_y=0



def clean_data(data):
    return data[~np.isnan(data)]

for j in range(len(classes_cutoffs)):
    binBoundaries = np.linspace(0,max_gain,10)
    counts, bins = np.histogram(clean_data(gain[:, j]), bins=binBoundaries)

    max_gain_y = max(max_gain_y, max(counts))

    binBoundaries = np.linspace(0,max_FP,10)
    counts, bins = np.histogram(clean_data(FP[:, j]), bins=binBoundaries)
    max_FPs_y = max(max_FPs_y, max(counts))
    print(max_FPs_y)

    binBoundaries = np.linspace(0,max_spec,10)
    counts, bins = np.histogram(clean_data(specificity[:, j]), bins=binBoundaries)
    max_spec_y = max(max_spec_y, max(counts))

    binBoundaries = np.linspace(0,max_f1,10)

    counts, bins = np.histogram(clean_data(f1_score[:, j]), bins=binBoundaries)
    max_f1_y = max(max_f1_y, max(counts))


plt.figure(figsize=(15, 5))
for j, class_cutoff in enumerate(classes_cutoffs):
    binBoundaries = np.linspace(0,max_gain,10)
    plt.subplot(1, 3, j+1)
    plt.hist(gain[:, j], bins=binBoundaries, alpha=0.7, label='Gain')
    plt.xlabel('Gain', fontsize=15)
    plt.ylabel('Zliczenia', fontsize=15)
    plt.title(f'Klasa odcięcia {class_cutoff}', fontsize=15)
    plt.xlim(0,max_gain)
    plt.ylim(0,max_gain_y+1)

plt.show()

plt.figure(figsize=(15, 5))
for j, class_cutoff in enumerate(classes_cutoffs):
    binBoundaries = np.linspace(0,max_FP,10)
    plt.subplot(1, 3, j+1)
    plt.hist(FP[:, j], bins=binBoundaries, alpha=0.7, label='FP')
    plt.xlabel('FP', fontsize=15)
    plt.ylabel('Zliczenia', fontsize=15)
    plt.title(f'Klasa odcięcia {class_cutoff}', fontsize=15)
    plt.xlim(0,max_FP)
    plt.ylim(0,max_FPs_y+1)

plt.show()

plt.figure(figsize=(15, 5))
for j, class_cutoff in enumerate(classes_cutoffs):
    binBoundaries = np.linspace(0,max_spec,10)
    plt.subplot(1, 3, j+1)
    plt.hist(specificity[:, j], bins=binBoundaries, alpha=0.7, label='Specificity')
    plt.xlabel('Specyficzność', fontsize=15)
    plt.ylabel('Zliczenia', fontsize=15)
    plt.title(f'Klasa odcięcia {class_cutoff}', fontsize=15)
    plt.xlim(0,max_spec)
    plt.ylim(0,max_spec_y+1)

plt.show()

plt.figure(figsize=(15, 5))
for j, class_cutoff in enumerate(classes_cutoffs):
    binBoundaries = np.linspace(0,max_f1,10)
    plt.subplot(1, 3, j+1)
    plt.hist(f1_score[:, j], bins=binBoundaries, alpha=0.7, label='F1')
    plt.xlabel('F1', fontsize=15)
    plt.ylabel('Zliczenia', fontsize=15)
    plt.title(f'Klasa odcięcia {class_cutoff}', fontsize=15)
    plt.xlim(0,max_f1)
    plt.ylim(0,max_f1_y+1)

plt.show()

#z algorytmu

i=0
for csv_file in csv_files:
    #gains_final, gains_std, FPs_final, FPs_std, specificities, specificities_std=valid.calculate_average_gain_and_FP(csv_file, val_size, classes_cutoffs, min_class, max_class, nr_of_splits)
    gains_csv, FPs_csv, specificities_csv, f1_csv, specificities_mod_csv =valid.calculate_average_gain_and_FP(csv_file, val_size, classes_cutoffs, min_class, max_class, nr_of_splits)
    all_gains[i*nr_of_splits:(i+1)*nr_of_splits]=gains_csv
    all_FPs[i*nr_of_splits:(i+1)*nr_of_splits]=FPs_csv
    all_specificities[i*nr_of_splits:(i+1)*nr_of_splits]=specificities_csv
    all_f1_scores[i*nr_of_splits:(i+1)*nr_of_splits]=f1_csv
    all_specificities_mod[i*nr_of_splits:(i+1)*nr_of_splits]=specificities_mod_csv

    #w tym momencie już nie jest wykorzystywane w nowym sposobie liczenia średniej
    '''
    gain_csv=np.nanmean(gains_csv, axis=0)
    gain_std_csv=np.nanstd(gains_csv, axis=0)
    FP_csv=np.nanmean(FPs_csv, axis=0)
    FP_std_csv=np.nanstd(FPs_csv, axis=0)
    specificity_csv=np.nanmean(specificities_csv, axis=0)
    specificity_std_csv=np.nanstd(specificities_csv, axis=0)
    f1_csv=np.nanmean(f1_csv, axis=0)
    f1_std_csv=np.nanstd(f1_csv, axis=0)
    
    print(gain_csv.shape, gain_std_csv.shape, FP_csv.shape, FP_std_csv.shape, specificity_csv.shape, specificity_std_csv.shape)
    
    
    gain[i]=gain_csv
    gain_std[i]=gain_std_csv
    FP[i]=FP_csv
    FP_std[i]=FP_std_csv
    specificity[i]=specificity_csv
    specificity_std[i]=specificity_std_csv
    f1_score[i]=f1_csv
    f1_std[i]=f1_std_csv
    '''
    i+=1
    print(gain.shape, gain_std.shape, FP.shape, FP_std.shape, specificity.shape, specificity_std.shape)
    print(all_gains.shape, all_FPs.shape, all_specificities.shape)
    
#tak było wcześniej
'''
gain_mean=np.nanmean(gain, axis=0)
gain_std_mean=np.nanmean(gain_std, axis=0)
FP_final_mean=np.nanmean(FP, axis=0)
FP_std_mean=np.nanmean(FP_std, axis=0)
specificity_mean=np.nanmean(specificity, axis=0)
specificity_std_mean=np.nanmean(specificity_std, axis=0)
f1_score_mean=np.nanmean(f1_score, axis=0)
f1_std_mean=np.nanmean(f1_std, axis=0)
'''

gain_mean=np.nanmean(all_gains, axis=0)
gain_std_mean=np.nanstd(all_gains, axis=0)
FP_final_mean=np.nanmean(all_FPs, axis=0)
FP_std_mean=np.nanstd(all_FPs, axis=0)
specificity_mean=np.nanmean(all_specificities, axis=0)
specificity_std_mean=np.nanstd(all_specificities, axis=0)
f1_score_mean=np.nanmean(all_f1_scores, axis=0)
f1_std_mean=np.nanstd(all_f1_scores, axis=0)
specificity_mod_mean=np.nanmean(all_specificities_mod, axis=0)
specificity_mod_std_mean=np.nanstd(all_specificities_mod, axis=0)

print("Gain", gain_mean, gain_std_mean)
print("FP", FP_final_mean, FP_std_mean)
print("Specificity", specificity_mean, specificity_std_mean)
print("F1", f1_score_mean, f1_std_mean)
print("Specificity_mod", specificity_mod_mean, specificity_mod_std_mean)

#plots
plt.figure(figsize=(10,5))
plt.errorbar(classes_cutoffs, gain_mean, yerr=gain_std_mean, fmt='o')
plt.xlabel('Odcięcie klasy', fontsize=15)
plt.ylabel('Gain', fontsize=15)
#plt.title('Gain for different classes cutoff')
plt.show()

plt.figure(figsize=(10,5))
plt.errorbar(classes_cutoffs, FP_final_mean, yerr=FP_std_mean, fmt='o')
plt.xlabel('Odcięcie klasy', fontsize=15)
plt.ylabel('FP', fontsize=15)
#plt.title('FP for different classes cutoff')
plt.show()

plt.figure(figsize=(10,5))
plt.errorbar(classes_cutoffs, specificity_mean, yerr=specificity_std_mean, fmt='o')
plt.xlabel('Odcięcie klasy', fontsize=15)
plt.ylabel('Specyficzność', fontsize=15)
#plt.title('Specificity for different classes cutoff')
plt.show()

plt.figure(figsize=(10,5))
plt.errorbar(classes_cutoffs, f1_score_mean, yerr=f1_std_mean, fmt='o')
plt.xlabel('Odcięcie klasy', fontsize=15)
plt.ylabel('F1', fontsize=15)
#plt.title('F1 for different classes cutoff')
plt.show()

plt.figure(figsize=(10,5))
plt.errorbar(classes_cutoffs, specificity_mod_mean, yerr=specificity_mod_std_mean, fmt='o')
plt.xlabel('Odcięcie klasy', fontsize=15)
plt.ylabel('Specyficzność zmodyfikowana', fontsize=15)
#plt.title('Specificity for different classes cutoff')
plt.show()


#max_value
max_gain=np.nanmax(all_gains)
max_FP=np.nanmax(all_FPs)
max_spec=np.nanmax(all_specificities)
max_f1=np.nanmax(all_f1_scores)
max_spec_mod=np.nanmax(all_specificities_mod)

max_gain_y = 0
max_FPs_y=0
max_spec_y=0
max_f1_y=0
max_spec_mod_y=0



# Najpierw obliczamy maksymalne wartości y na wszystkich wykresach
for j in range(len(classes_cutoffs)):
    binBoundaries = np.linspace(0,max_gain,10)
    counts, bins = np.histogram(clean_data(all_gains[:, j]), bins=binBoundaries)
    max_gain_y = max(max_gain_y, max(counts))

    binBoundaries = np.linspace(0,max_FP,10)
    counts, bins = np.histogram(clean_data(all_FPs[:, j]), bins=binBoundaries)
    max_FPs_y = max(max_FPs_y, max(counts))

    binBoundaries = np.linspace(0,max_spec,10)
    counts, bins = np.histogram(clean_data(all_specificities[:, j]), bins=binBoundaries)
    max_spec_y = max(max_spec_y, max(counts))

    binBoundaries = np.linspace(0,max_f1,10)
    counts, bins = np.histogram(clean_data(all_f1_scores[:, j]), bins=binBoundaries)
    max_f1_y = max(max_f1_y, max(counts))

    binBoundaries = np.linspace(0,max_spec_mod,10)
    counts, bins = np.histogram(clean_data(all_specificities_mod[:, j]), bins=binBoundaries)
    max_spec_mod_y = max(max_spec_mod_y, max(counts))

plt.figure(figsize=(15, 5))
for j, class_cutoff in enumerate(classes_cutoffs):
    binBoundaries = np.linspace(0,max_gain,10)
    plt.subplot(1, 3, j+1)
    plt.hist(all_gains[:, j], bins=binBoundaries, alpha=0.7, label='Gain')
    plt.xlabel('Gain', fontsize=15)
    plt.ylabel('Zliczenia', fontsize=15)
    plt.title(f'Klasa odcięcia {class_cutoff}', fontsize=15)
    plt.xlim(0,max_gain)
    plt.ylim(0,max_gain_y+1)

plt.show()

plt.figure(figsize=(15, 5))
for j, class_cutoff in enumerate(classes_cutoffs):
    binBoundaries = np.linspace(0,max_FP,10)
    plt.subplot(1, 3, j+1)
    plt.hist(all_FPs[:, j], bins=binBoundaries, alpha=0.7, label='FP')
    plt.xlabel('FP', fontsize=15)
    plt.ylabel('Zliczenia', fontsize=15)
    plt.title(f'Klasa odcięcia {class_cutoff}', fontsize=15)
    plt.xlim(0,max_FP)
    plt.ylim(0,max_FPs_y+1)

plt.show()

plt.figure(figsize=(15, 5))
for j, class_cutoff in enumerate(classes_cutoffs):
    binBoundaries = np.linspace(0,max_spec,10)
    plt.subplot(1, 3, j+1)
    plt.hist(all_specificities[:, j], bins=binBoundaries, alpha=0.7, label='Specificity')
    plt.xlabel('Specyficzność', fontsize=15)
    plt.ylabel('Zliczenia', fontsize=15)
    plt.title(f'Klasa odcięcia {class_cutoff}', fontsize=15)
    plt.xlim(0,max_spec)
    plt.ylim(0,max_spec_y+1)

plt.show()

plt.figure(figsize=(15, 5))
for j, class_cutoff in enumerate(classes_cutoffs):
    binBoundaries = np.linspace(0,max_f1,10)
    plt.subplot(1, 3, j+1)
    plt.hist(all_f1_scores[:, j], bins=binBoundaries, alpha=0.7, label='F1')
    plt.xlabel('F1', fontsize=15)
    plt.ylabel('Zliczenia', fontsize=15)
    plt.title(f'Klasa odcięcia {class_cutoff}', fontsize=15)
    plt.xlim(0,max_f1)
    plt.ylim(0,max_f1_y+1)

plt.show()

plt.figure(figsize=(15, 5))
for j, class_cutoff in enumerate(classes_cutoffs):
    binBoundaries = np.linspace(0,max_spec_mod,10)
    plt.subplot(1, 3, j+1)
    plt.hist(all_specificities_mod[:, j], bins=binBoundaries, alpha=0.7, label='Specificity_mod')
    plt.xlabel('Specyficzność zmodyfikowana', fontsize=15)
    plt.ylabel('Zliczenia', fontsize=15)
    plt.title(f'Klasa odcięcia {class_cutoff}', fontsize=15)
    plt.xlim(0,max_spec_mod)
    plt.ylim(0,max_spec_mod_y+1)

plt.show()