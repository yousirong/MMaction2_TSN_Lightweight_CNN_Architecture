# Check Pytorch installation
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMAction2 installation
import mmaction
print(mmaction.__version__)

# Check MMCV installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())


# colab에서 TPU사용해보기 
# import tensorflow as tf
# import os

# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])

# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)

# strategy = tf.distribute.TPUStrategy(resolver)

# 내장 gpu 사용
# import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from mmaction.apis import inference_recognizer, init_recognizer
from mmengine import Config
from mmengine.runner import set_random_seed
import os
import pandas as pd  # Import pandas module
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import logging
from operator import itemgetter
from tqdm import tqdm
import csv  # Import csv module
import numpy as np

Log_Format = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename="abnormal_logfile_resnet.log", filemode="w", format=Log_Format, level=logging.DEBUG)
logger = logging.getLogger()

cfg = Config.fromfile('./configs/recognition/tsn/juneyong_backbones/tsn_r50_1x1x8_20e_Dassult.py')
cfg.dataset_type = 'VideoDataset'
cfg.data_root = './datasets/allData/'
cfg.test_dataloader.dataset.type = 'VideoDataset'
# # TL 테스트 데이터 어노테이션 txt
# cfg.test_dataloader.dataset.ann_file = './datasets/allData/TL_test_abnormal_normal.txt'
# # 유튜브 실제 데이터 어노테이션 txt
cfg.test_dataloader.dataset.ann_file = './datasets/allData/Youtube_abnormal.txt'


cfg.test_dataloader.dataset.data_prefix.video = './datasets/abnormal_data_test'
cfg.setdefault('omnisource', False)
# cfg.model.cls_head.num_classes = 2
cfg.load_from = './work_dirs/tsn_r50_1x1x8_20e_Dassult/best_acc_top1_epoch_100.pth'
cfg.work_dir = './work_dirs/tsn_r50_1x1x8_20e_Dassult/'
cfg.test_dataloader.videos_per_gpu = 12
cfg.optim_wrapper.optimizer.lr = cfg.optim_wrapper.optimizer.lr / 8 / 16
cfg.total_epochs = 120
cfg.default_hooks.checkpoint.interval = 5
cfg.default_hooks.logger.interval = 5
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.evaluation.save_best='auto'

# Setup a checkpoint file to load
checkpoint = './work_dirs/tsn_r50_1x1x8_20e_Dassult/best_acc_top1_epoch_100.pth'
model = init_recognizer(cfg, checkpoint, device='cuda:0')


csv_filename = './datasets/Youtube_abnormal.csv'
res_path = './datasets/allData/'

# Read video paths from CSV file
df = pd.read_csv(csv_filename)
file_list_path = df['Video Path'].tolist()
#1(abnormal 폭행임 true) 와 0(normal 폭행아님 false)  
hand_pose = {"normal": "0", "assault": "1"}

gt_list = []
predict_list = []

i=1
for file in tqdm(file_list_path):
    # Check if the file has a non-zero size
    if os.path.getsize(file) == 0:
        print(f"Skipping empty file: {file}")
        continue

    for pose in hand_pose:
        if pose in file:
            print("ground truth : ", pose)
            break

    video = file
    label = './datasets/abnormal_test.txt'
    results = inference_recognizer(model, video)
    # # 모델 예측 점수 중 'driver_assault' 레이블에 대한 점수 추출
    # assault_score = results.pred_score[2].item()  # 'driver_assault' 레이블의 인덱스는 2라고 가정
    # print(assault_score)
    # # 폭행 여부에 따라 레이블 결정
    # if assault_score > 0.5:  # 임계값을 0.5로 설정 (필요에 따라 조정 가능)
    #     prediction = "assault"
    # else:
    #     prediction = "normal"
    scores = results.pred_score.cpu().numpy()
    print("Scores", scores)
    
    # Use argmax to find the index of the highest score
    predicted_index = np.argmax(scores)
    print("predicted_index", predicted_index)
    prediction = "assault" if predicted_index == 1 else "normal"  # Assuming index 1 corresponds to 'assault'
    # print("predicted_index_scores", predicted_index)
    print("Predicted: ", prediction)
    logger.debug(str(i) + "\t" + file + "\t" + pose + "\t" + prediction)
    gt_list.append(pose)
    predict_list.append(prediction)
    

    i += 1
    
    
    
 