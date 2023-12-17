import os
import shutil

file_path_list = "local_invalid.txt"

folders_to_delete = set()

with open(file_path_list, 'r') as file:
    for line in file:
        path_parts = line.strip().split('/')
        folder_to_delete = '/'.join(path_parts[:-1])  # 마지막 부분을 제외한 나머지 경로
        folders_to_delete.add(folder_to_delete)
        print(folder_to_delete)

for folder_path in folders_to_delete:
    print(f"삭제될 폴더: {folder_path}")
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path, ignore_errors=True)