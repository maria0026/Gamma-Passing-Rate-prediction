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

#print("wersja", tf.__version__)
reference_folder = '/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference'
evaluation_folder = '/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/evaluation'
txt_folder = '/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/txt'

reference_files = os.listdir(reference_folder)
evaluation_files = os.listdir(evaluation_folder)
txt_files = os.listdir(txt_folder)

#znajdowanie dcm bez par
def find_dcm_without_pair(reference_files, evaluation_files):
    files_without_pair = []
    i=0
    for file in reference_files:
        pair_name = file.replace("Predicted", "Portal")
        if pair_name not in evaluation_files:
            files_without_pair.append(file)
            i+=1

    for file in evaluation_files:
        pair_name = file.replace("Portal", "Predicted")
        if pair_name not in reference_files:
            files_without_pair.append(file)
            i+=1

    print("liczba dcm bez pary", i) #825
    return files_without_pair

files_without_pair=find_dcm_without_pair(reference_files, evaluation_files)

#creating dataframe for our model
def create_df(reference_files, evaluation_files, txt_files):
    file_pairs = []
    for file in reference_files:
        pair_name = file.replace("Predicted", "Portal")
        txt_name = file.replace("Predicted-Dose-", "").replace(".dcm", ".txt").replace("  ", " ")
        if pair_name in evaluation_files and txt_name in txt_files:
            reference_path = os.path.join(reference_folder, file)  #Join with reference_folder
            evaluation_path = os.path.join(evaluation_folder, pair_name)  #Join with evaluation_folder
            txt_path = os.path.join(txt_folder, txt_name)

            file_pairs.append((reference_path, evaluation_path, txt_path)) 

    df_pairs = pd.DataFrame(file_pairs, columns=['ref', 'eval', 'txt'])
    df_pairs.to_csv('file_pairs.txt', sep='\t', index=False)
    return df_pairs

df=create_df(reference_files, evaluation_files, txt_files)
print(df)
df[['ref', 'eval', 'txt']].to_csv('ref_eval2.csv', sep='\t', index=False)

'''
ref_dist = pydicom.dcmread(df.iloc[0,0])
ref_dist = ref_dist.pixel_array
print("Reference shape:", ref_dist.shape)

eval_dist = pydicom.dcmread(df.iloc[0,1])
eval_dist = eval_dist.pixel_array
print("Evaluation shape:",eval_dist.shape)

txt= df.iloc[0,2]
print("Txt:", txt)
'''

def calculate_gamma(ref, eval, txt, dose_percent_threshold=2, distance_mm_threshold=2, lower_percent_dose_cutoff=7, draw=False) :

    ref_dist_dcm = pydicom.dcmread(ref)
    
    intercept_ref=ref_dist_dcm[(0x0028, 0x1052)].value
    slope_ref=ref_dist_dcm[(0x0028, 0x1053)].value
    ref_dist = ref_dist_dcm.pixel_array
    ref_size=ref_dist.shape
    print("ref_dist shape", ref_size)

    #ref_dist=resize(ref_dist,ref_size)

    eval_dist_dcm = pydicom.dcmread(eval)
    intercept_eval=eval_dist_dcm[(0x0028, 0x1052)].value
    slope_eval=eval_dist_dcm[(0x0028, 0x1053)].value
    exposure_sequence = eval_dist_dcm[(0x3002, 0x0030)].value
    #Get the Meterset Exposure value from the first item in the sequence
    meterset_exposure = exposure_sequence[0][(0x3002, 0x0032)].value
    eval_dist=eval_dist_dcm.pixel_array
    eval_size=eval_dist.shape
    print("eval_dist shape", eval_size)

    #print("max ref", np.max(ref_dist), "max eval", np.max(eval_dist))
    if meterset_exposure is not None and slope_ref is not None and intercept_ref is not None:
        ref_dist=(ref_dist*slope_ref+intercept_ref)*(meterset_exposure)
    else:
        return 0, 0
    
    if slope_eval is not None and intercept_eval is not None:
        eval_dist=(eval_dist*slope_eval+intercept_eval)
    else:
        return 0, 0
    #print(np.max(eval_dist)/np.max(ref_dist))
    #eval_dist = resize(eval_dist, ref_size)  
    #print("max ref", np.max(ref_dist), "max eval", np.max(eval_dist))

    with open(txt, 'r') as file:    

        lines = file.readlines()
        #Iterate through each line
        for line in lines:
            #Check if the line contains "Passed Area Gamma <1.0"
            if "Area Gamma < 1.0"  in line:
                # Split the line by whitespace to extract the value
                parts = line.split()
                #print(parts)
                gamma_txt = float(parts[5])
                break


    #pixel_rescale=0.336*(1024/1190)
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
        'interp_fraction': 15,  # Should be 10 or more for more accurate results
        'max_gamma': 1.1,
        'random_subset': None,
        'local_gamma': False,
        'ram_available': 2**29  # 1/2 GB
    }
    
    start_time = time.time()
    # Calculate the gamma passing rate
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

    return pass_ratio, gamma_txt

def calculate_gamma_for_df(df, dose_percent_threshold, distance_mm_threshold, lower_percent_dose_cutoff):
    pass_ratios = []
    gamma_txts = []

    for index, row in df.iterrows():
        ref = row['ref']
        eval = row['eval']
        txt = row['txt']
        print(ref, eval, txt)
        
        pass_ratio, gamma_txt = calculate_gamma(ref, eval, txt, dose_percent_threshold=dose_percent_threshold, distance_mm_threshold=distance_mm_threshold, lower_percent_dose_cutoff=lower_percent_dose_cutoff)
        print("Pass ratio:", pass_ratio, "Gamma txt:", gamma_txt)
        pass_ratios.append(pass_ratio)
        gamma_txts.append(gamma_txt)

    # Add the calculated values to the original DataFrame
    df['pass_ratio'] = pass_ratios
    df['gamma_txt'] = gamma_txts

    print(df)
    #export df['pass_ratio', 'gamma_txt'] to csv
    df[['pass_ratio','gamma_txt']].to_csv('gamma_results.csv', sep='\t', index=False)

#calculate_gamma_for_df(df, dose_percent_threshold=2, distance_mm_threshold=2, lower_percent_dose_cutoff=10)

df_train, df_test= train_test_split(df, test_size=0.2, random_state=42)

#resize, then normalization
def load_and_preprocess_images(file_paths, desired_size):
    images=np.zeros((len(file_paths), desired_size[0], desired_size[1]), dtype='uint8')
    
    for i, file_path in enumerate(file_paths):
        #Load DICOM image
        dicom_data = pydicom.dcmread(file_path)
        image = dicom_data.pixel_array
        #preprocessing
        #image = resize(image, desired_size)  
        #normalize
        
        image=image.astype(np.float32) / image.max()
        images[i,:,:]=image
        print(i)
    return images


#X_processed = load_and_preprocess_images(df['X'], ref_size)
#print(X_processed.shape)

'''
# Define a function to apply to each row of the DataFrame
def calculate_gamma_for_row(row):
    ref = row['ref']
    eval = row['eval']
    txt = row['txt']
    pass_ratio, gamma_txt = calculate_gamma(ref, eval, txt)
    
    return pass_ratio, gamma_txt

# Apply the function to each row of the DataFrame
df[['pass_ratio', 'gamma_txt']] = df.apply(calculate_gamma_for_row, axis=1, result_type='expand')

eval='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/evaluation/4845686-Portal-Dose-GARDLO SROD.-1-plan1.dcm'
eval_dist_dcm = pydicom.dcmread(eval)
print(eval_dist_dcm)

intercept_eval=eval_dist_dcm[(0x0028, 0x1052)].value
slope_eval=eval_dist_dcm[(0x0028, 0x1053)].value
exposure_sequence = eval_dist_dcm[(0x3002, 0x0030)].value
#Get the Meterset Exposure value from the first item in the sequence
meterset_exposure = exposure_sequence[0][(0x3002, 0x0032)].value
'''



#sieć- obrazek liczba

'''
ref1='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference/18482-Predicted-Dose-MOSTEK-2-plan1.dcm'
ref2='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference/38179-Predicted-Dose-eKRTAN-2-plan1.dcm'
ref3='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference/9246129-Predicted-Dose-eKRTAN-1-plan1.dcm'
ref4='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference/27211-Predicted-Dose-eSKORA-1-plan1.dcm'
ref5='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference/18482-Predicted-Dose-MOSTEK-1-plan1.dcm'

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

'''
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#skalowanie
train_images = train_images / 255.0
test_images = test_images / 255.0

'''