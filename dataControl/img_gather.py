# import os
# import shutil

# def collect_mp4_files(source_dir, video_dir):
#     if not os.path.exists(video_dir):
#         os.makedirs(video_dir)
    
#     # source_dir에서 mp4 파일을 찾음
#     for root, dirs, files in os.walk(source_dir):
#         for file in files:
#             if file.endswith('.mp4'):
#                 source_file = os.path.join(root, file)
#                 video_file = os.path.join(video_dir, file)
                
#                 # 파일을 새 위치로 복사
#                 shutil.copy2(source_file, video_file)
#                 print(f"Copied: {source_file} to {video_file}")

# source_directory = './TS7'  # mp4 파일이 있는 소스 폴더 경로
# video_directory = './video'  # mp4 파일을 저장할 타겟 폴더 경로

# collect_mp4_files(source_directory, video_directory)



# def delete_files_to_fit_quota(directory, max_size):
#     # 폴더의 파일들과 그들의 사이즈를 가져옴
#     files_and_sizes = [(f, os.path.getsize(os.path.join(directory, f))) for f in os.listdir(directory) if f.endswith('.mp4')]
    
#     # 파일들을 사이즈에 따라 정렬
#     files_and_sizes.sort(key=lambda x: x[1], reverse=True)
    
#     # 현재 폴더 사이즈 계산
#     current_size = sum(size for _, size in files_and_sizes)
    
#     # 파일들을 삭제하여 폴더 사이즈를 max_size 이하로 만듦
#     for file, size in files_and_sizes:
#         if current_size <= max_size:
#             break
#         os.remove(os.path.join(directory, file))
#         current_size -= size
#         print(f"Deleted {file}, freed {size} bytes, remaining size {current_size} bytes")

# 사용 예시
# video_directory = './video'  # mp4 파일이 있는 폴더 경로
# max_folder_size = 0.01 * 1024 * 1024 * 1024  # 예: 2GB 제한

# delete_files_to_fit_quota(video_directory, max_folder_size)

# import os
# import shutil

# def delete_folder(folder_path):
#     """ 폴더와 그 안의 모든 내용을 삭제하는 함수 """
#     for file in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, file)
#         if os.path.isfile(file_path) or os.path.islink(file_path):
#             os.unlink(file_path)
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)
#     os.rmdir(folder_path)

# base_path = "C:/Users/user/Documents/vscode_project/HUFS_savingDriver/TS5"  # 상위 폴더 경로를 여기에 입력하세요

# # 상위 폴더 내의 모든 하위 폴더를 탐색
# for folder in os.listdir(base_path):
#     if folder.startswith("SGA210") and os.path.isdir(os.path.join(base_path, folder)):
#         subfolder_path = os.path.join(base_path, folder)

#         # 하위 폴더 내의 모든 폴더를 탐색
#         for subfolder in os.listdir(subfolder_path):
#             full_subfolder_path = os.path.join(subfolder_path, subfolder)
#             if os.path.isdir(full_subfolder_path):
#                 video_folder_path = os.path.join(subfolder_path, subfolder, "video")
#                 img_folder_path = os.path.join(subfolder_path, subfolder, "img")

#             # 'video' 폴더 내의 모든 파일을 찾아 상위 폴더로 이동
#             if os.path.exists(video_folder_path):
#                 for file in os.listdir(video_folder_path):
#                     if file.endswith(".mp4"):  # 비디오 파일 확장자를 여기에 맞게 조정하세요
#                         file_path = os.path.join(video_folder_path, file)
#                         shutil.move(file_path, subfolder_path)
                
#                 # 파일 이동 후 'video' 폴더 삭제
#                 if not os.listdir(video_folder_path):
#                     os.rmdir(video_folder_path)

#             # 'img' 폴더 삭제
#             if os.path.exists(img_folder_path):
#                 delete_folder(img_folder_path)

#             # 비디오 및 이미지 폴더가 포함된 하위 폴더 삭제
#             if os.path.exists(full_subfolder_path):
#                 delete_folder(full_subfolder_path)

import os
import shutil

def delete_folder(folder_path):
    """ 폴더와 그 안의 모든 내용을 삭제하는 함수 """
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    os.rmdir(folder_path)

num = 7

base_path = f"C:/Users/user/Documents/vscode_project/HUFS_savingDriver/TS{num}"  # 상위 폴더 경로를 여기에 입력하세요

# 상위 폴더 내의 모든 하위 폴더를 탐색
for folder in os.listdir(base_path):
    if folder.startswith("SGA210") and os.path.isdir(os.path.join(base_path, folder)):
        subfolder_path = os.path.join(base_path, folder)

        # 하위 폴더 내의 모든 폴더를 탐색
        for subfolder in os.listdir(subfolder_path):
            full_subfolder_path = os.path.join(subfolder_path, subfolder)
            if os.path.isdir(full_subfolder_path):
                video_folder_path = os.path.join(subfolder_path, subfolder, "video")
                img_folder_path = os.path.join(subfolder_path, subfolder, "img")

            # 'video' 폴더 내의 모든 파일을 찾아 상위 폴더로 이동
            if os.path.exists(img_folder_path):
                for file in os.listdir(img_folder_path):
                    if file.endswith(".jpg"):  # 비디오 파일 확장자를 여기에 맞게 조정하세요
                        file_path = os.path.join(img_folder_path, file)
                        shutil.move(file_path, full_subfolder_path)
                delete_folder(img_folder_path)
                delete_folder(video_folder_path)