# kakaopay-hw

## 문서
- [`실행 방법 문서`](./Documents/01_실행_방법_문서.md)(model 학습 및 단계별 설명)
- [`데이터 속성별 탐색 과정 및 집값 예측 모형 개발 단계별 고려 사항을 포함한 문서`](./Documents/02_데이터_속성별_탐색_과정_및_집값_예측_모형_개발_단계별_고려_사항_문서.md)
- [`예측 모델에 대한 성능 평가 문서`](./Documents/03_예측_모델에_대한_성능_평가_문서.md)
- 코드 목록
    - [`01_Convert_CSV_to_ARFF.ipynb`](./work/01_Convert_CSV_to_ARFF.ipynb)  
        - csv파일을 arff파일로 변환하고 train, test 데이터 분리 진행
    - [`02_Feature_Selection.ipynb`](./work/02_Feature_Selection.ipynb)
        - 특징 분포 출력 그리고 특징간의 산점도 그림 출력
        - CFS 기반의 특징선택 알고리즘 적용
    - [`03_Predict_Model.ipynb`](./work/03_Predict_Model.ipynb)
        - 모델 학습 그리고 모델 평가 진행
        - 사용한 모델: 선형회귀, 랜덤 포레스트, SMOreg (회귀를 위한 SVM, support vector machine 알고리즘)
    - [`04_Final_Model.ipynb`](./work/04_Final_Model.ipynb)
        - 최종 모델: SMOreg (+ RBF kernel)
    - 기타: [`wrapper`](./work/wrapper) 모듈  
      Weka API를 편히 쓰기 위해 만든 모듈

## 필요한 소프트웨어
- `docker` :whale:
- `docker-compose`

### 실행 환경
- Python 3.6 (conda) :snake:
- jupyter notebook :notebook:

### 실행 환경 만들기
> ***시간이 꽤 소요됩니다.*** <img src="/assets/kakaopay-emoji.png" width=32/>
- docker-compose
```bash
git clone https://github.com/jazz4rabbit/kakaopay-hw
cd kakaopay-hw
docker-compose up -d --build
```  
> 이후, 웹브라우저에 `127.0.0.1:1088`입력, 비밀번호: `muzi`입력하면, 코드를 실행해 볼 수 있습니다.
- 1088포트를 사용중인 경우 port 설정필요 (기본 값: `1088`)  
  `.env` 파일에서 `PORT`에 해당하는 port 숫자로 변경

#### 사용한 라이브러리
<img src="https://www.cs.waikato.ac.nz/ml/images/Weka%20(software)%20logo.png" alt="Weka logo"> &nbsp;&nbsp;&nbsp;&nbsp; <img src="https://fracpete.github.io/python-weka-wrapper/_static/logo.jpg" alt="Weka logo">
- [python-weka-wrapper](https://fracpete.github.io/python-weka-wrapper/index.html) 
