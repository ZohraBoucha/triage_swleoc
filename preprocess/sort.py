import shutil
import os

# Specify the directory you want to iterate through
directory_path = 'datasets/dataset_dara/clinics'

# Iterate through all files in the directory
for root, dirs, files in os.walk(directory_path):
    for file in files:
        print('file: ', file)
        name_arr = file.split(' ')
        if(len(name_arr)<2):
            print('lesser: ', file)
            continue

        pid = name_arr[1]
        print('pid: ', pid)

        PATH = f'datasets/cleaned_dataset_dara/{pid}'
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        file_path = os.path.join(root, file)
        shutil.copy(file_path, PATH)

        print(f"File found: {file_path}")

