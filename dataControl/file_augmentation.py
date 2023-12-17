from pathlib import Path
import shutil
import os
import glob


def copy_and_reorder_images(folder, subfolder):
    folder_path = Path(folder)
    jpg_files = list(folder_path.glob("*IMG000*.jpg"))
    if len(jpg_files) < 5:
        return "Less than 5 images found in " + str(folder)

    order = [1, 2, 3, 4, 5, 4, 3, 2]
    new_files = []
    original_files = set()

    for i, num in enumerate(order, start=1):
        source_file = next((f for f in jpg_files if f.stem.endswith(str(num))), None)
        if not source_file:
            return f"Image {num} not found in " + str(folder)

        new_file_name = folder_path / f"{subfolder}IMG000_{i}.jpg"
        new_files.append(new_file_name)

        shutil.copy(source_file, new_file_name)
        original_files.add(source_file)
    
    for file in original_files:
        os.remove(file)

def check_file_count(path):
    file_count = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
    if file_count < 5:
        shutil.rmtree(path)

def rename_files_in_folder(folder_path, new_name_pattern):
    for i, filename in enumerate(os.listdir(folder_path)):
        old_file = os.path.join(folder_path, filename)
        
        new_file = os.path.join(folder_path, f"{new_name_pattern}_{i+1}.jpg")

        os.rename(old_file, new_file)

for number in range(5, 8):           
    base_path = f"C:/Users/user/Documents/vscode_project/HUFS_savingDriver/TS{number}"

    for folder in os.listdir(base_path):
        if folder.startswith("SGA210") and os.path.isdir(os.path.join(base_path, folder)):
            subfolder_path = os.path.join(base_path, folder)

            for subfolder in os.listdir(subfolder_path):
                full_subfolder_path = os.path.join(subfolder_path, subfolder)
                rename_files_in_folder(full_subfolder_path, "IMG000")
                check_file_count(full_subfolder_path)
                copy_and_reorder_images(full_subfolder_path, subfolder)