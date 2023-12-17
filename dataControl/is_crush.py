import os
import shutil

num = 5
base_path = f"C:/Users/user/Documents/vscode_project/HUFS_savingDriver/OriginalData/TS{num}"

for folder in os.listdir(base_path):
    if folder.startswith("SGA210") and os.path.isdir(os.path.join(base_path, folder)):
        subfolder_path = os.path.join(base_path, folder)

        # 하위 폴더 내의 모든 폴더를 탐색
        for subfolder in os.listdir(subfolder_path):
            full_subfolder_path = os.path.join(subfolder_path, subfolder)
            if os.path.isdir(full_subfolder_path):
                video_folder_path = os.path.join(subfolder_path, subfolder, "video")
                img_folder_path = os.path.join(subfolder_path, subfolder, "img")

            if os.path.exists(img_folder_path):
                for file in os.listdir(img_folder_path):
                    full_img_path = os.path.join(img_folder_path, file)
                    if os.path.getsize(full_img_path) == 0:
                        print(f"파일이 비어있습니다 (0바이트): {full_img_path}")