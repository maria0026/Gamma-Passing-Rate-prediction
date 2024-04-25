import normalize
import numpy as np
import torch
import os
import shutil
import pydicom
from PIL import Image

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
    
def split_folder(df,source_path,desired_column):

    for index, row in df.iterrows():
        file_path = row[desired_column]
        print(file_path)
        file_path=file_path.replace("  ", " ")
        filename=os.path.basename(file_path)
        destination_folder = os.path.join(source_path, str(row['class']))
        destination_path = os.path.join(destination_folder, filename)

        os.makedirs(destination_folder, exist_ok=True)

        #Check if the source file exists before attempting to move it
        if os.path.exists(file_path):
            #Move the file
            shutil.move(file_path, destination_path)
            print(f"Moved {file_path} to {destination_path}")
        else:
            print(f"Source file {file_path} does not exist")

'''
def preprocess_dicom_to_jpg(dicom_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(dicom_dir):
        if filename.endswith('.dcm'):
            file_path=os.path.join(dicom_dir, filename)
            array_dist,size=normalize.normalize_ref_image(file_path)
            print(array_dist.shape)
            image = Image.fromarray(array_dist)
            image = image.convert('L')
            output_path = os.path.join(output_dir, filename.replace('.dcm', '.jpg'))
            image.save(output_path)

def preprocess_folder(source_path):
    folders=os.listdir(source_path)
    #print(folders)
    for folder in folders:
        folder=os.path.join(source_path, folder)
        #print(folder)
        output_dir=folder+'_jpg'
        preprocess_dicom_to_jpg(folder, output_dir)

#resize, then normalization
def load_and_preprocess_ref_images(file_paths, desired_size):
    #images=np.zeros((len(file_paths), desired_size[0], desired_size[1]), dtype='uint8')
    num_images=len(file_paths)
    images = torch.zeros((num_images, 1, desired_size[0], desired_size[1]), dtype=torch.uint8)
    
    for i, file_path in enumerate(file_paths):
        ref_dist, ref_size=normalize.normalize_ref_image(file_path)
        print(np.max(ref_dist))
        if ref_size==desired_size:
        #image=image.astype(np.float32) / image.max()
            #images[i,:,:]=ref_dist.astype('uint8')
            ref_dist = ref_dist.reshape((1, desired_size[0], desired_size[1]))
            images[i, :, :, :] = torch.tensor(ref_dist, dtype=torch.uint8)
            print(i)
        else:
            pass
    return images

def load_and_preprocess_eval_images(file_paths, desired_size):
    images=np.zeros((len(file_paths), desired_size[0], desired_size[1]), dtype='uint8')
    
    for i, file_path in enumerate(file_paths):
        if isinstance(file_path, str):
            eval_dist, eval_size=normalize.normalize_eval_image(file_path)
            #print(np.max(eval_dist))
            #preprocessing
            #image = resize(image, desired_size)
            if eval_size==desired_size:
                images[i,:,:]=eval_dist.astype('uint8') #? co z tym datatype
                print(np.max(images[i,:,:]))
                print(i)
            else:
                pass  
        else: 
            images[i,:,:]=np.zeros((desired_size[0], desired_size[1]), dtype='float16')

    return images
'''

