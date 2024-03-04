import os
import shutil
data_folder='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/data_final'

#chcemy z tego folderu odfiltrować tylko dcm oraz gdy występuje słowo Predicted dać do folderu referal, a Portal do evalutaion

#Filter files by extension and keyword
all_files = os.listdir(data_folder)
extension = 'dcm'
predicted = 'Predicted'
reference_files = [file for file in all_files if file.endswith(extension) and predicted in file]
print(reference_files)

portal='Portal'
evaluation_files = [file for file in all_files if file.endswith(extension) and portal in file]
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