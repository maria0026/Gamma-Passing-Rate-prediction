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

#there are folders with reference, evaluation distributions and txt files
#distibution is a dcm with summary of dose received by patient in the treatment plan
#reference- distribution with predicted dose, filename format: ID-Predicted-Dose-NAME-X-planY.dcm
#evaluation- distribution with portal dose- measured, filename format: ID-Portal-Dose-NAME-X-planY.dcm
#txt- text file containing information about Gamma Passing Rate

reference_folder = '/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference'
evaluation_folder = '/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/evaluation'
txt_folder = '/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/txt'

reference_files = os.listdir(reference_folder)
evaluation_files = os.listdir(evaluation_folder)
txt_files = os.listdir(txt_folder)

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

    print("liczba dcm bez pary", i) 
    return files_without_pair

files_without_pair=find_dcm_without_pair(reference_files, evaluation_files)
#there are some files (825) without a pair, we will not take them into account

#creating dataframe with file paths containing only pairs
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

def normalize_image(image, reference=True, evaluation=False):
    #Load DICOM image
    dicom_data = pydicom.dcmread(image)
    #dicom structure:
    #(0028, 1052) Rescale Intercept
    #(0028, 1053) Rescale Slope                
    intercept = dicom_data[(0x0028, 0x1052)].value
    slope = dicom_data[(0x0028, 0x1053)].value
    array_dist = dicom_data.pixel_array
    size = array_dist.shape
    
    if reference and slope is not None and intercept is not None:
        array_dist = (array_dist * slope + intercept)
        return array_dist, slope, intercept, size
    
    elif evaluation and slope is not None and intercept is not None:
        exposure_sequence = dicom_data[(0x3002, 0x0030)].value
        meterset_exposure = exposure_sequence[0][(0x3002, 0x0032)].value
        if meterset_exposure is not None:
            array_dist = (array_dist * slope + intercept) / meterset_exposure
            return array_dist, slope, intercept, meterset_exposure, size
        else:
            return array_dist, slope, intercept, None, size
    else:
        return array_dist, None, None, None, size

        #dist = (dist * slope + intercept) / meterset_exposure



def calculate_gamma(ref, eval, txt, dose_percent_threshold=2, distance_mm_threshold=2, lower_percent_dose_cutoff=7, draw=False) :
    
    ref_dist, slope_ref, intercept_ref, ref_size=normalize_image(ref, reference=True, evaluation=False)
    eval_dist, slope_eval, intercept_eval, meterset_exposure, eval_size=normalize_image(eval, reference=False, evaluation=True)

    #print("max ref", np.max(ref_dist), "max eval", np.max(eval_dist))

    if meterset_exposure is None or slope_ref is None or intercept_ref is None:
        return 0, 0
    
    if slope_eval is None or intercept_eval is None:
        return 0, 0
    #print(np.max(eval_dist)/np.max(ref_dist))
    #eval_dist = resize(eval_dist, ref_size)  
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

#first attempt- give only reference and gamma passing rate as input
#read data_for_nn.csv and create a dataframe
df = pd.read_csv('data_for_nn.csv', sep=',')
print(df)

#resize, then normalization
def load_and_preprocess_images(file_paths, desired_size, reference=True, evaluation=False):
    images=np.zeros((len(file_paths), desired_size[0], desired_size[1]), dtype='uint8')
    
    if reference:

        for i, file_path in enumerate(file_paths):
            ref_dist, slope_ref, intercept_ref, ref_size=normalize_image(file_path, reference=True, evaluation=False)
            if ref_size==desired_size:
            #image=image.astype(np.float32) / image.max()
                images[i,:,:]=ref_dist.astype('float16')
                print(i)
            else:
                pass

    if evaluation:
        for i, file_path in enumerate(file_paths):
            eval_dist, slope_eval, intercept_eval, meterset_exposure, eval_size=normalize_image(file_path, reference=False, evaluation=True)
            #preprocessing
            #image = resize(image, desired_size)  
            images[i,:,:]=eval_dist.astype('float16')
            print(i)

    return images

def map_to_class(gamma_value):
    if gamma_value < 95:
        return 0
    elif 95 <= gamma_value < 96:
        return 1
    elif 96 <= gamma_value < 96.5:
        return 2
    elif 96.5 <= gamma_value < 97:
        return 3
    elif 97 <= gamma_value < 98:
        return 4
    elif 98 <= gamma_value < 99:
        return 5
    elif 99 <= gamma_value < 99.5:
        return 6
    elif 99.5 <= gamma_value < 99.8:
        return 7
    elif 99.8 <= gamma_value < 99.9:
        return 8
    elif 99.9 <= gamma_value <= 100:
        return 9
    else:
        return None


df['class'] = df['gamma_txt'].apply(map_to_class)
print(df['class'])
df_train, df_test= train_test_split(df, test_size=0.2, random_state=42)
desired_size=(1024,1024)
X_train = load_and_preprocess_images(df_train['ref'], desired_size)
Y_train=df_train['class']
X_test = load_and_preprocess_images(df_test['ref'], desired_size)
Y_test=df_test['class']
#print(X_train.shape)

#sieć- obrazek liczba
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=desired_size),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10)
test_loss, test_acc = model.evaluate(X_train,  Y_train, verbose=2)

print('\nTest accuracy:', test_acc)


'''
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
