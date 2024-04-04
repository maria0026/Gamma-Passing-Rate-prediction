import normalize
import numpy as np

#resize, then normalization
def load_and_preprocess_ref_images(file_paths, desired_size):
    images=np.zeros((len(file_paths), desired_size[0], desired_size[1]), dtype='uint8')
    
    for i, file_path in enumerate(file_paths):
        ref_dist, ref_size=normalize.normalize_ref_image(file_path)
        if ref_size==desired_size:
        #image=image.astype(np.float32) / image.max()
            images[i,:,:]=ref_dist.astype('float16')
            print(i)
        else:
            pass
    return images

def load_and_preprocess_eval_images(file_paths, desired_size):
    images=np.zeros((len(file_paths), desired_size[0], desired_size[1]), dtype='uint8')
    
    for i, file_path in enumerate(file_paths):
        if isinstance(file_path, str):
            eval_dist, eval_size=normalize.normalize_eval_image(file_path)
            #preprocessing
            #image = resize(image, desired_size)
            if eval_size==desired_size:
                images[i,:,:]=eval_dist.astype('float16')
                print(i)
            else:
                pass  
        else: 
            images[i,:,:]=np.zeros((desired_size[0], desired_size[1]), dtype='uint8')

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