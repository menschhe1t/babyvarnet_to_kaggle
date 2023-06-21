# 2023 baby unet
2023 SNU FastMRI challenge

## 1. 폴더 계층

### 폴더의 전체 구조
![image](https://github.com/LISTatSNU/FastMRI_challenge/assets/39179946/b551e277-4134-41bb-9d1a-8275a65c1eb7)
* FastMRI_challenge, Data, result 폴더가 위의 구조대로 설정되어 있어야 default argument를 활용할 수 있습니다.
* 본 github repository는 FastMRI_challenge 폴더입니다.
* Data 폴더는 MRI data 파일을 담고 있으며 아래에 상세 구조를 첨부하겠습니다.
* result 폴더는 학습한 모델의 weights을 기록하고 validation, leaderboard dataset의 reconstruction image를 저장하는데 활용되며 아래에 상세 구조를 첨부하겠습니다.

### Data 폴더의 구조
![image](https://github.com/LISTatSNU/FastMRI_challenge/assets/39179946/6e3b4ed1-b027-4b09-a0b1-3d10cb51b93a)
* train, val:
    * train, val 폴더는 각각 모델을 학습(train), 검증(validation)하는데 사용하며 각각 image, kspace 폴더로 나뉩니다.
    * 참가자들은 generalization과 representation의 trade-off를 고려하여 train, validation의 set을 자유로이 나눌 수 있습니다.
    * image와 kspace 폴더에 들어있는 파일의 형식은 다음과 같습니다: brain_{mask 형식}_{순번}.h5
    * ex) brain_acc8_3.h5  
    * {mask 형식}은 "acc4"과 "acc8" 중 하나입니다.
    * {순번}은 1 ~ 181 사이의 숫자입니다. 
* Leaderboard:
   * **Leaderboard는 성능 평가를 위해 활용하는 dataset이므로 절대로 학습 과정에 활용하면 안됩니다.**
   * Leaderboard 폴더는 mask 형식에 따라서 acc4과 acc8 폴더로 나뉩니다.
   * acc4과 acc8 폴더는 각각 image, kspace 폴더로 나뉩니다.
   * image와 kspace 폴더에 들어있는 파일의 형식은 다음과 같습니다: brain_test_{순번}.h5
   * {순번}은 1 ~ 58 사이의 숫자입니다. 

### result 폴더의 구조
![image](https://github.com/LISTatSNU/FastMRI_challenge/assets/39179946/1174e5bf-1551-4dcc-8b6a-77d3fd63fb12)
* result 폴더는 모델의 이름에 따라서 여러 폴더로 나뉠 수 있습니다.
* 위 그림에서는 default argument인 test_Unet만 고려했습니다. 
* test_Unet 폴더는 아래 3개의 폴더로 구성되어 있습니다.
  * checkpoints - model.pt, best_model.pt의 정보가 있습니다. 모델의 weights 정보를 담고 있습니다.
  * reconstructions_val - validation dataset의 reconstruction을 저장합니다. brain_{mask 형식}_{순번}.h5 형식입니다. (```train.py``` 참고)
  * reconstructions_forward - Leaderboard dataset의 reconstruction을 저장합니다. brain_test_{순번}.h5 형식입니다. (```evaluation.py``` 참고)

## 2. 폴더 정보
```bash
├── .gitignore
├── evaluate.py
├── leaderboard_eval.py
├── plot.py
├── README.md
├── train.py
└── utils
│   ├── common
│   │   ├── loss_function.py
│   │   └── utils.py
│   ├── data
│   │   ├── load_data.py
│   │   └── transforms.py
│   ├── learning
│   │   ├── test_part.py
│   │   └── train_part.py
│   └── model
│       └── unet.py
├── Data
└── result
```

## 3. Before you start
* ```train.py```, ```evaluation.py```, ```leaderboard_eval.py``` 순으로 코드를 실행하면 됩니다.
* ```train.py```
   * train/validation을 진행하고 학습한 model의 결과를 result 폴더에 저장합니다.
   * 가장 성능이 좋은 모델의 weights을 ```best_model.pt```으로 저장합니다. 
* ```evaluation.py```
   * ```train.py```으로 학습한 ```best_model.pt```을 활용해 leader_board dataset을 reconstruction하고 그 결과를 result 폴더에 저장합니다.
   * acc4와 acc8 옵션을 활용해 두개의 샘플링 마스크(4X, 8X)에 대해서 전부 reconstruction을 실행합니다.
* ```leaderboard_eval.py```
   * ```evaluation.py```을 활용해 생성한 reconstruction의 SSIM을 측정합니다.
   * acc4와 acc8 옵션을 활용해 두개의 샘플링 마스크(4X, 8X)에 대해서 전부 측정을 합니다.

## 4. How to set?
```bash
conda create -n baby_unet python=3.9
conda activate baby_unet

pip3 install numpy
pip3 install torch
pip3 install h5py
pip3 install scikit-image
pip3 install opencv-python
pip3 install matplotlib
```

## 5. How to train?
```bash
python3 train.py
```

## 6. How to reconstruct?
```bash
python3 evaluate.py -m acc4
```

```bash
python3 evaluate.py -m acc8
```

## 7. How to evaluate LeaderBoard Dataset?
```bash
python3 leaderboard_eval.py -m acc4
```

```bash
python3 leaderboard_eval.py -m acc8
```

## 8. Plot!
```bash
python3 plot.py 
```
![image](https://github.com/LISTatSNU/FastMRI_challenge/assets/39179946/22dea43d-db54-42c4-9054-1b1ea461c648)
