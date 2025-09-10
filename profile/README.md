
# 🏥 Wound Vision (상처 관리 AI 솔루션)

![Wound vision (2)_page-0001](https://github.com/user-attachments/assets/37a633c9-0bba-4284-a199-3ed2f75a2770)



Wound Vision은 상처 사진을 AI로 분석하고 분류하여, 신속한 초기 대응을 위한 관리 방법과 주변 의료 정보를 제공하는 서비스입니다. 사용자는 이 서비스를 통해 상처 관리에 필요한 정보를 얻고, 피부 질환의 악화와 확산을 예방할 수 있습니다.

<br>

## 💡 프로젝트 배경 (Motivation)

일상에서 가벼운 상처가 발생했을 때, 우리는 병원에 가야 할지, 집에서 관리해야 할지 판단하기 어렵습니다. 온라인에는 신뢰하기 어려운 정보가 넘쳐나고, 경증 환자가 응급실에 몰려 의료 시스템에 과부하를 주기도 합니다.

Wound Vision은 이러한 현실적인 문제들을 해결하고자 시작되었습니다.

- **의료 접근성 문제**: 경증 환자의 불필요한 응급실 방문을 줄여 의료 시스템의 효율성을 높입니다.
- **정보 격차 문제**: 부정확한 온라인 정보 속에서 신뢰할 수 있는 초기 대응 가이드를 제공합니다.
- **디지털 헬스케어 기회**: 비약적으로 발전하는 AI 기술과 비대면 의료 서비스 시장의 성장을 기회로 삼았습니다.

<br>

## ✨ 주요 기능 (Features)

### 📸 AI 상처 분석
- 사용자가 상처 부위를 촬영하여 업로드하면, AI가 이미지를 분석하여 상처 유형(여드름, 멍, 화상 등)을 분류합니다.

### 📝 맞춤 관리 팁 제공
- 분석된 상처 유형에 따라, 집에서 할 수 있는 올바른 응급처치 및 관리 방법을 단계별로 제공합니다.

### 🏥 주변 병원 및 약국 추천
- 사용자 위치를 기반으로, 상처 유형에 맞는 전문 병원이나 가까운 약국의 위치 정보를 지도 위에 표시해 줍니다.

<br>

## ⚙️ 기술 스택 (Tech Stack)

- **AI & Machine Learning**: ResNet101, DeepLabV3, Microsoft Azure Custom Vision
- **Service & UI**: Gradio
- **APIs**: Kakao Map API, Google Maps API
- **Data Source**: Kaggle, AI Hub

<br>

## 📊 아키텍처 및 개발 프로세스 (Architecture & Process)

![Wound vision (2)_page-0007](https://github.com/user-attachments/assets/a9896e4e-5524-4668-8a49-d950ccfbfba6)
![Wound vision (2)_page-0008](https://github.com/user-attachments/assets/a4949f86-eb4d-4421-87e3-043028256808)



## 🤖 AI 모델 개발 과정 (AI Model Development)

정확도 높은 상처 분류 모델을 개발하기 위해 다음과 같은 데이터 전처리 과정을 거쳤습니다.

### 1. 데이터 증강 (Data Augmentation)

한정된 데이터셋을 효과적으로 활용하기 위해 기존 이미지에 밝기, 감마, 회전 등 다양한 변형을 주어 데이터의 양을 증강시켰습니다. 이를 통해 모델이 다양한 조건의 이미지에 대응할 수 있도록 학습 성능을 높였습니다.

```python
# 이미지 밝기 조정 코드 예시
def adjust_brightness(image, value):
    """이미지 밝기 조정"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v + value, 0, 255).astype(np.uint8)
    adjusted_hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)
```

### 2. 데이터 크롭핑 (Data Cropping)

DeepLabV3 모델을 사용하여 이미지에서 배경을 제거하고, 상처가 있는 신체 부위만 정확히 추출(Cropping)했습니다. 이를 통해 AI가 불필요한 정보 없이 오직 상처 부위에만 집중하여 학습할 수 있도록 했습니다.

```python
# DeepLabV3를 이용한 배경 제거 및 크롭핑 코드 예시
def remove_background(image):
    """DeepLabV3 이용해 신체 부위 분할 후 배경 제거 및 크롭"""
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    model.eval()
    # ... (중략)
    mask = (output_predictions == 15).astype(np.uint8) * 255
    coords = cv2.findNonZero(mask)
    # ... (후략)
```

### 3. 모델 학습 (Model Training)

전처리가 완료된 데이터는 **Microsoft Azure Custom Vision**을 통해 라벨링 및 학습을 진행했습니다.

![Wound vision (2)_page-0010](https://github.com/user-attachments/assets/ce76749a-f3e5-4dd1-92da-e9bf5755db29)


<br>

## 📈 모델 성능 및 결과 (Results)

![Wound vision (2)_page-0011](https://github.com/user-attachments/assets/2638fe9f-713f-4880-8d18-fe3012ab66dd)

| 메트릭 | 전처리 전 | 전처리 후 | 개선율 |
|--------|-----------|-----------|--------|
| **정밀도 (Precision)** | 93.5% | 99.4% | +5.9% |
| **재현율 (Recall)** | 90.3% | 99.2% | +8.9% |
| **F1-Score** | 91.8% | 99.3% | +7.5% |

특히 정밀도(Precision)는 93.5%에서 99.4%로, 재현율(Recall)은 90.3%에서 99.2%로 크게 상승하며 모델의 신뢰도를 확보했습니다.

<br>

## 📱 서비스 화면 (Demo)

![Wound vision (2)_page-0012](https://github.com/user-attachments/assets/037904ea-b9db-470e-a96d-33024bb17a45)



[![Wound Vision Demo Video](https://img.youtube.com/vi/7Rr6NqAYDx0/maxresdefault.jpg)](https://youtu.be/7Rr6NqAYDx0)
*클릭하여 Wound Vision의 실제 동작 과정을 확인해보세요*


<br>

## 🚀 향후 계획 (Future Work)

### 🔬 AI 기술 고도화
- 식약처 의료기기 소프트웨어 허가 취득 추진
- 다양한 상처 유형 데이터셋 추가 확보 (10만+ 이미지) 및 모델 성능 고도화
- 엣지 컴퓨팅 기술을 적용하여 개인정보 보호 강화

### 💼 사업 모델 다각화
- B2C, B2B, B2G 등 다중 사업 모델 구축
- 의료기관과의 파트너십 확대 및 원격 의료 플랫폼 연동
- 해외 진출을 통한 글로벌 시장 개척

### 🛡️ 신뢰성 및 법규 준수
- 의료법 및 개인정보보호법 완전 준수 체계 구축
- 의료진 다중 검증을 통한 고품질 서비스 제공
- ISO 13485 (의료기기 품질관리) 인증 취득

<br>


