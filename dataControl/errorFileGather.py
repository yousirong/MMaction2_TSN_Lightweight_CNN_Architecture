import os
import shutil

# 경로를 읽을 txt 파일
file_list_path = 'invalid_jpg_files_new.txt'
updated_file_list_path = 'local_invalid.txt'

# 변경할 경로와 새 경로
old_path_part = './datasets/allData/Img_AIhub'
new_path_part = 'C:/Users/user/Documents/vscode_project/HUFS_savingDriver'

# 변경된 경로를 저장할 리스트
updated_paths = []

line_count = 0

# 파일 경로를 읽고 변경
with open(file_list_path, 'r') as file:
    for line in file:
        file_path = line.strip()
        updated_file_path = file_path.replace(old_path_part, new_path_part)
        updated_paths.append(updated_file_path)

# 변경된 경로를 같은 파일 또는 새 파일에 저장
with open(updated_file_list_path, 'w') as file:  # 같은 파일에 덮어쓸 경우
    for path in updated_paths:
        file.write(path + '\n')

# 새로운 폴더 생성
new_folder_path = 'collected_files'
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

# 파일 경로를 읽고 새 폴더로 복사
with open(updated_file_list_path, 'r') as file:
    for line in file:
        file_path = line.strip()  # 줄바꿈 문자 제거
        if os.path.exists(file_path):
            # 파일 크기 확인
            if os.path.getsize(file_path) > 0:
                shutil.copy(file_path, new_folder_path)
            else:
                line_count += 1
                print(f"파일이 비어있습니다 (0바이트): {file_path}")
        else:
            print(f"파일이 존재하지 않습니다: {file_path}")

print(line_count)