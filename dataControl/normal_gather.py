import pandas as pd
import os
import shutil
import json

def delete_folder(folder_path):
    """ 폴더와 그 안의 모든 내용을 삭제하는 함수 """
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    os.rmdir(folder_path)

for num in range(5, 8):
    base_path = f"C:/Users/user/Documents/vscode_project/HUFS_savingDriver/TS{num}"
    assault = f"./assault_label_{num}.csv"

    df = pd.read_csv(assault)
    df['occupants_info'] = df['occupants_info'].apply(lambda x: json.loads(x.replace("'", '@').replace('"', "'").replace("@", '"')))
    df_normal = df[(df['action_2'] == '운전하다') | (df['action_2'] == '무언가를보다') | (df['action_2'] == '차량문열기')]
    df_values = df_normal.loc[:, 'img_id'].values

    # 정상 데이터 끄집어내기
    for folder in os.listdir(base_path):
        if folder.startswith("SGA210") and os.path.isdir(os.path.join(base_path, folder)):
            subfolder_path = os.path.join(base_path, folder)

            for subfolder in os.listdir(subfolder_path):
                full_subfolder_path = os.path.join(subfolder_path, subfolder)

                if os.path.exists(full_subfolder_path):
                    for imgfile in os.listdir(full_subfolder_path):
                        if imgfile not in df_values:
                            os.remove(os.path.join(full_subfolder_path, imgfile))

                if os.path.exists(full_subfolder_path):
                    for imgfile in os.listdir(full_subfolder_path):
                        if imgfile.endswith(".jpg"):
                            file_path = os.path.join(full_subfolder_path, imgfile)
                            shutil.move(file_path, subfolder_path)
                os.rmdir(full_subfolder_path)
        if os.path.isdir(subfolder_path) and not os.listdir(subfolder_path):
            os.rmdir(subfolder_path)

    # 정상 데이터 폴더로 묶기
    for folder in os.listdir(base_path):
        if folder.startswith("SGA210") and os.path.isdir(os.path.join(base_path, folder)):
            subfolder_path = os.path.join(base_path, folder)
            img_files = [f for f in os.listdir(subfolder_path) if f.endswith('.jpg')]
            
            for i in range(0, len(img_files), 5):
                group = img_files[i:i + 5]

                new_folder_path = os.path.join(subfolder_path, f'group_{i // 5 + 1}')
                os.makedirs(new_folder_path, exist_ok=True)

                for img in group:
                    shutil.move(os.path.join(subfolder_path, img), new_folder_path)
