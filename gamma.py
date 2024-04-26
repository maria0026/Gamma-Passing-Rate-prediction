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
    '''
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
    '''
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

    #export df['pass_ratio', 'gamma_txt'] to csv
    return df
    

