import pymedphys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pydicom
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from skimage import exposure
import time
import cv2
from skimage.registration import phase_cross_correlation
import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.layers import LeakyReLU
import normalize

#print("wersja", tf.__version__)

#there are folders with reference, evaluation distributions and txt files
#distibution is a dcm with summary of dose received by patient in the treatment plan
#reference- distribution with predicted dose, filename format: ID-Predicted-Dose-NAME-X-planY.dcm
#evaluation- distribution with portal dose- measured, filename format: ID-Portal-Dose-NAME-X-planY.dcm
#txt- text file containing information about Gamma Passing Rate

    
def draw_maps(ref_dist, eval_dist, gamma):
    # Plot reference distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(ref_dist, cmap='viridis')
    plt.title('Reference Distribution')
    plt.colorbar()

    # Plot evaluation distribution
    plt.subplot(1, 3, 2)
    plt.imshow(eval_dist, cmap='viridis')
    plt.title('Evaluation Distribution')
    plt.colorbar()

    # Plot gamma distribution
    plt.subplot(1, 3, 3)
    plt.imshow(gamma, cmap='coolwarm', vmin=0, vmax=1)
    plt.title('Gamma Passing Rate')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def draw_histogram(valid_gamma, gamma_options, pass_ratio, x_ref, y_ref):
    num_bins = (
        gamma_options['interp_fraction'] * gamma_options['max_gamma'])
    bins = np.linspace(0, gamma_options['max_gamma'], int(num_bins) + 1)

    plt.hist(valid_gamma, bins, density=True)
    #if density is True, y value is probability density; otherwise, it is count in a bin
    plt.xlim([0, gamma_options['max_gamma']])
    plt.xlabel('gamma index')
    plt.ylabel('probability density')

    if gamma_options['local_gamma']:
        gamma_norm_condition = 'Local gamma'
    else:
        gamma_norm_condition = 'Global gamma'

    plt.title(f"Dose cut: {gamma_options['lower_percent_dose_cutoff']}% | {gamma_norm_condition} ({gamma_options['dose_percent_threshold']}%/{gamma_options['distance_mm_threshold']}mm) | Pass Rate(\u03B3<=1): {pass_ratio*100:.2f}% \n ref pts: {len(y_ref)*len(x_ref)} | valid \u03B3 pts: {len(valid_gamma)}")
    plt.show()
    #plt.savefig('gamma_hist.png', dpi=300)


def calculate_gamma(ref, eval, txt, dose_percent_threshold=2, distance_mm_threshold=2, lower_percent_dose_cutoff=5, draw=False) :
    
    ref_dist, ref_size=normalize.normalize_ref_image(ref)
    eval_dist, eval_size=normalize.normalize_eval_image(eval)

    if ref_dist is None or eval_dist is None:
        return None, None
    #rezise eval_dist to ref_dist size
    eval_dist = resize(eval_dist, ref_size)
    eval_size=eval_dist.shape
    shift = cv2.phaseCorrelate(ref_dist, eval_dist)
    #shift= phase_cross_correlation(ref_dist, eval_dist)
    print(shift)
    # Apply the shift to the shifted image
    rows, cols = eval_dist.shape
    M = np.float32([[1, 0, round(shift[0][0])], [0, 1, round(shift[0][1])]])
    print(M)
    #M= np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    eval_dist = cv2.warpAffine(eval_dist, M, (cols, rows))
    print(np.max(ref_dist), np.max(eval_dist))

    #print("max ref", np.max(ref_dist), "max eval", np.max(eval_dist))

    #print(np.max(eval_dist)/np.max(ref_dist))

    #print("max ref", np.max(ref_dist), "max eval", np.max(eval_dist))

    with open(txt, 'r') as file:   
         
        lines = file.readlines()
        #Iterate through each line
        for line in lines:
            #Check if the line contains Passed/Failed "Area Gamma < 1.0"
            if "Area Gamma < 1.0"  in line:
                #Split the line by whitespace to extract the value
                parts = line.split()
                #structure in txt file: Passed	Area Gamma < 1.0	99.8 %	95.0 %, we need sixth element to get gamma passing rate
                gamma_txt = float(parts[5])
                break

    x_ref=np.linspace(0,400, ref_size[0])
    y_ref=np.linspace(0,400, ref_size[1])
    x_eval = np.linspace(0, 400, eval_size[0]) 
    y_eval = np.linspace(0, 400, eval_size[1]) 

    axes_reference = (x_ref,y_ref)
    axes_evaluation = (x_eval,y_eval)

    dose_reference = ref_dist
    dose_evaluation = eval_dist

    gamma_options = {
        'dose_percent_threshold': dose_percent_threshold,
        'distance_mm_threshold': distance_mm_threshold,
        'lower_percent_dose_cutoff': lower_percent_dose_cutoff,
        'interp_fraction': 15,  #Should be 10 or more for more accurate results
        'max_gamma': 1.1,
        'random_subset': None,
        'local_gamma': False,
        'ram_available': 2**29  #1/2 GB
    }
    
    start_time = time.time()
    #Calculate the gamma passing rate
    gamma = pymedphys.gamma(
        axes_reference,
        dose_reference,
        axes_evaluation,
        dose_evaluation,
        **gamma_options
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    #print(elapsed_time)
    valid_gamma = gamma[~np.isnan(gamma)]
    pass_ratio = np.sum(valid_gamma < 1) / len(valid_gamma)
    pass_ratio=pass_ratio*100
    pass_ratio=round(pass_ratio, 3)

    if draw:
        draw_maps(ref_dist, eval_dist, gamma)
        draw_histogram(valid_gamma, gamma_options, pass_ratio, x_ref, y_ref)

    return pass_ratio, gamma_txt

def calculate_gamma_for_df(df, dose_percent_threshold, distance_mm_threshold, lower_percent_dose_cutoff, draw=False):
    pass_ratios = []
    gamma_txts = []

    for index, row in df.iterrows():
        ref = row['ref']
        eval = row['eval']
        txt = row['txt']
        print(ref, eval, txt)
        
        pass_ratio, gamma_txt = calculate_gamma(ref, eval, txt, dose_percent_threshold=dose_percent_threshold, distance_mm_threshold=distance_mm_threshold, lower_percent_dose_cutoff=lower_percent_dose_cutoff, draw=draw)
        print("Pass ratio:", pass_ratio, "Gamma txt:", gamma_txt)
        pass_ratios.append(pass_ratio)
        gamma_txts.append(gamma_txt)

    # Add the calculated values to the original DataFrame
    df['pass_ratio'] = pass_ratios
    df['gamma_txt'] = gamma_txts

    print(df)
    #export df['pass_ratio', 'gamma_txt'] to csv
    df[['pass_ratio','gamma_txt']].to_csv('gamma_results5.csv', sep='\t', index=False)




ref='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference/270144-Predicted-Dose-e_MIEDNICA-3-plan4.dcm'
eval='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/evaluation/270144-Portal-Dose-e_MIEDNICA-3-plan4.dcm'
txt='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/txt/270144-e_MIEDNICA-3-plan4.txt'

ref='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference/4160526-Predicted-Dose-CZERNIAK-2-plan1.dcm'
eval='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/evaluation/4160526-Portal-Dose-CZERNIAK-2-plan1.dcm'
txt='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/txt/4160526-CZERNIAK-2-plan1.txt'
'''
def shift_image(ref_dist, eval_dist):
    ref_dist, ref_slope, ref_intercept, ref_size =normalize.normalize_image(ref_dist, reference=True, evaluation=False)
    eval_dist, eval_slope, eval_intercept, meterset_exposure, eval_size =normalize.normalize_image(eval_dist, reference=False, evaluation=True)

    # Find the shift using phase correlation
    shift = cv2.phaseCorrelate(ref_dist, eval_dist)

    # Apply the shift to the shifted image
    rows, cols = eval_dist.shape
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    eval_dist_shifted = cv2.warpAffine(eval_dist, M, (cols, rows))

    return ref_dist, eval_dist_shifted

'''
#ref='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference/8045699-Predicted-Dose-eMIGDALEK-2-plan1.dcm'
#eval='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/evaluation/8045699-Portal-Dose-eMIGDALEK-2-plan1.dcm'
#txt='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/txt/8045699-eMIGDALEK-2-plan1.txt'

#ref='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference/12524-Predicted-Dose-PIERS LW WW-2-plan1.dcm'
#eval='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/evaluation/12524-Portal-Dose-PIERS LW WW-2-plan1.dcm'
#txt='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/txt/12524-PIERS LW WW-2-plan1.txt'

#ref='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference/884040-Predicted-Dose-BLIZNA PW SO-4-plan2.dcm'
#eval='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/evaluation/884040-Portal-Dose-BLIZNA PW SO-4-plan2.dcm'
#txt='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/txt/884040-BLIZNA PW SO-4-plan2.txt'
'''
dicom_data = pydicom.dcmread(ref)
print(dicom_data)
pass_ratio, gamma_txt=calculate_gamma(ref, eval, txt, dose_percent_threshold=2, distance_mm_threshold=2, lower_percent_dose_cutoff=5, draw=True)
print(pass_ratio, gamma_txt)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
'''
'''
ref1='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference/18482-Predicted-Dose-MOSTEK-2-plan1.dcm'
ref2='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference/38179-Predicted-Dose-eKRTAN-2-plan1.dcm'
ref3='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference/9246129-Predicted-Dose-eKRTAN-1-plan1.dcm'
ref4='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference/27211-Predicted-Dose-eSKORA-1-plan1.dcm'
ref5='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference/18482-Predicted-Dose-MOSTEK-1-plan1.dcm'
eval='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/evaluation/4845686-Portal-Dose-GARDLO SROD.-1-plan1.dcm'
#98,7
eval1='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/evaluation/18482-Portal-Dose-MOSTEK-2-plan1.dcm'
#99,6
eval2='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/evaluation/38179-Portal-Dose-eKRTAN-2-plan1.dcm'
#99.8
eval3='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/evaluation/9246129-Portal-Dose-eKRTAN-1-plan1.dcm'
#99.9
eval4='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/evaluation/27211-Portal-Dose-eSKORA-1-plan1.dcm'
#100
eval5='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/evaluation/18482-Portal-Dose-MOSTEK-1-plan1.dcm'

txt1='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/txt/18482-MOSTEK-2-plan1.txt'
txt2='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/txt/38179-eKRTAN-2-plan1.txt'
txt3='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/txt/9246129-eKRTAN-1-plan1.txt'
txt4='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/txt/27211-eSKORA-1-plan1.txt'
txt5='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/txt/18482-MOSTEK-1-plan1.txt'

gamma1, gamma1_txt=calculate_gamma(ref1, eval1, txt1, dose_percent_threshold=2, distance_mm_threshold=2, lower_percent_dose_cutoff=10)
gamma2, gamma2_txt=calculate_gamma(ref2, eval2, txt2, dose_percent_threshold=2, distance_mm_threshold=2, lower_percent_dose_cutoff=10)
gamma3, gamma3_txt=calculate_gamma(ref3, eval3, txt3, dose_percent_threshold=2, distance_mm_threshold=2, lower_percent_dose_cutoff=10)
gamma4, gamma4_txt=calculate_gamma(ref4, eval4, txt4, dose_percent_threshold=2, distance_mm_threshold=2, lower_percent_dose_cutoff=10)
gamma5, gamma5_txt=calculate_gamma(ref5, eval5, txt5, dose_percent_threshold=2, distance_mm_threshold=2, lower_percent_dose_cutoff=10)
print("Txt:", gamma1_txt, gamma2_txt, gamma3_txt, gamma4_txt, gamma5_txt)
print("Gammy: ", gamma1, gamma2, gamma3, gamma4, gamma5)
'''
