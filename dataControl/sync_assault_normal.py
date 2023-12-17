import os
import shutil

def align_subfolder_count(folder1, folder2):
    subfolders1 = os.listdir(folder1)
    subfolders2 = os.listdir(folder2)

    min_count = min(len(subfolders1), len(subfolders2))

    for subfolder in subfolders1[min_count:]:
        shutil.rmtree(os.path.join(folder1, subfolder))

    for subfolder in subfolders2[min_count:]:
        shutil.rmtree(os.path.join(folder2, subfolder))

def main():
    assault = "C:/Users/user/Documents/vscode_project/HUFS_savingDriver/assault"
    normal = "C:/Users/user/Documents/vscode_project/HUFS_savingDriver/normal"

    assault_paths = {}
    normal_paths = {}
    for i in range(5, 8):
        key = f"TS{i}"
        assault_paths[key] = f"C:/Users/user/Documents/vscode_project/HUFS_savingDriver/assault/TS{i}"
        normal_paths[key] = f"C:/Users/user/Documents/vscode_project/HUFS_savingDriver/normal/TS{i}"

    align_subfolder_count(assault_paths["TS5"], normal_paths["TS5"])
    align_subfolder_count(assault_paths["TS6"], normal_paths["TS6"])
    align_subfolder_count(assault_paths["TS7"], normal_paths["TS7"])

    for ts in ["TS5", "TS6", "TS7"]:
        assault_path = assault_paths[ts]
        normal_path = normal_paths[ts]

        align_subfolder_count(assault_path, normal_path)

        assault_subfolders = os.listdir(assault_path)
        normal_subfolders = os.listdir(normal_path)

        # 하위 폴더 개수 맞추기
        for i in range(len(assault_subfolders)):
            align_subfolder_count(os.path.join(assault_path, assault_subfolders[i]), 
                                  os.path.join(normal_path, normal_subfolders[i]))



if __name__ == "__main__":
    main()