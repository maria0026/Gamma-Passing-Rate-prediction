import os
import shutil
import pandas as pd
import gamma

#chcemy z folderu data odfiltrować tylko dcm oraz gdy występuje słowo Predicted dać do folderu reference, a Portal do evalutaion
def data_sorting(data_folder, extension, first_keyword, second_keyword):

    all_files = os.listdir(data_folder)
    
    reference_files = [file for file in all_files if file.endswith(extension) and first_keyword in file]
    print(reference_files)
    
    evaluation_files = [file for file in all_files if file.endswith(extension) and second_keyword in file]
    print(evaluation_files)

    text_files = [file for file in all_files if file.endswith('.txt')]

    #Move filtered files to the destination folder
    for file in reference_files:
        source_path = os.path.join('data/data_final', file)
        destination_path = os.path.join('data/reference', file)
        shutil.move(source_path, destination_path)

    for file in evaluation_files:
        source_path = os.path.join('data/data_final', file)
        destination_path = os.path.join('data/evaluation', file)
        shutil.move(source_path, destination_path)    

    for file in text_files:
        source_path = os.path.join('data/data_final', file)
        destination_path = os.path.join('data/txt', file)
        shutil.move(source_path, destination_path)


def find_dcm_without_pair(reference_folder, evaluation_folder):
    reference_files = os.listdir(reference_folder)
    evaluation_files = os.listdir(evaluation_folder)

    files_without_pair = []
    
    for file in reference_files:
        pair_name = file.replace("Predicted", "Portal")
        if pair_name not in evaluation_files:
            files_without_pair.append(file)

    for file in evaluation_files:
        pair_name = file.replace("Portal", "Predicted")
        if pair_name not in reference_files:
            files_without_pair.append(file)

    return files_without_pair


#creating dataframe with file paths containing only pairs
def create_df(reference_folder, evaluation_folder, txt_folder):
    reference_files = os.listdir(reference_folder)
    evaluation_files = os.listdir(evaluation_folder)
    txt_files = os.listdir(txt_folder)
    file_pairs = []
    for file in reference_files:
        pair_name = file.replace("Predicted", "Portal")
        pair_name = pair_name.replace(",", "")
        txt_name = file.replace("Predicted-Dose-", "").replace(".dcm", ".txt").replace("  ", " ")
        txt_name = txt_name.replace(",", "")
        if pair_name in evaluation_files and txt_name in txt_files:
            reference_path = os.path.join(reference_folder, file)  #Join with reference_folder
            evaluation_path = os.path.join(evaluation_folder, pair_name)  #Join with evaluation_folder
            txt_path = os.path.join(txt_folder, txt_name)

            file_pairs.append((reference_path, evaluation_path, txt_path)) 

    df_pairs = pd.DataFrame(file_pairs, columns=['ref', 'eval', 'txt'])
    #df_pairs.to_csv('file_pairs.txt', sep='\t', index=False)
    return df_pairs



