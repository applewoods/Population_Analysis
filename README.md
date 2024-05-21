# 인구구조 기반 도시 유형화

## 1. 선행연구
### - 인구감소지역
- 행정안전부에서 5년 단위로 인구감소지역 지정하며, 2023년 9월 기준 89개의 도시가 인구감소지역으로 선정
- 인구감소지수는 총 8가지의 지표로 측정 됨</br>①연평균인구증감률, ②인구밀도, ③청년순이동률, ④주간인구, ⑤고령화 비율, ⑥유소년 비율, ⑦조출생률, ⑧재정자립도
- 참고: 행정안전부 인구감소지역 지정 [[바로가기](https://www.mois.go.kr/frt/sub/a06/b06/populationDecline/screen.do)]

<p align= 'center'>
    <figure align= 'center'>
        <img src='./img/행안부_인구감소지역_202309.jpg' width= '250px' title='인구감소지역'></img>
        <figcaption>그림1. 행안부 인구감소지역</figcaption>
    </figure>
</p>

### - 인구구조 및 인구이동 데이터
- 선행연구 조사를 통해 인구구조 및 인구이동 연령층 선정
- 총 5개의 연령층과 외국인 인구를 본 연구에서 사용</br>①신생아(0세), ②유소년인구(1세~14세), ③청년가임인구(20세~39세), ④소비활력인구(40세~59세), ⑤고령인구(65세 이상)

<p align= 'center'>
    <figure align= 'center'>
        <img src='./img/인구구분_선행연구.png', title= '인구 연령구분 선행연구'>
        <figcaption>그림2. 인구 연령구분 선행연구</figcaption>
    </figure>
</p>

## 2. 데이터
- 모든 데이터는 국가통계포털(KOSIS)에서 수집
- (공간범위) 전국 229개 시군구의 인구 시계열 데이터를 기반으로 도시유형 분류 수행
- (시간범위) 2013년부터 2022년까지 전연령 인구 데이터

구분 | 데이터 | 연령기준
:-: | :-: | :-:
1 | 신생아 비율 | 0세
2 | 유소년인구 비율 | 1세 ~ 14세
3 | 청년가임인구 비율 | 20세 ~ 39세
4 | 소비활력인구 비율 | 40세 ~ 59세
5 | 고령인구 비율 | 65세 이상
6 | 외국인 비율 | -

## 3. 분석 방법론
### 3.1. 데이터 전처리
- 시계열 인구 데이터를 분석에 그대로 사용할 경우 도시유형이 도시의 규모로 묶이는 경향이 있어 각 도시의 인구비율로 데이터를 전처리하여 분석 진행
- 추가로 MinMax Scaler를 사용하여 인구비율 데이터를 추가로 전처리 진행

### 3.2. TimeSeries K-Means Clustering
- Elbow Plot과 Shillhouette Plot 결과를 바탕으로 최적의 K 도출
- 시계열의 거리 측정 방식에는 대표적으로 유클리드 거리(Euclidean Distance)와 동적 시간 왜곡(Dynamic Time Warping; DTW)가 있으며, 본 연구에서는 시계열의 비슷한 패턴을 바탕으로 도시유형화를 하기 위해 DTW를 사용

<p align= 'center'>표1. Euclidean Distance와 Dynamic Time Warping(DTW)의 차이</p>

구분 | 유클리드거리</br>(Euclidean Distance) | 동적 시간 왜곡</br>(Dynamic Time Warping)
:-: | :-: | :-:
정의 | 시계열 각 지점의 거리의 합을 계산 | 시계열의 비선형적인 매핑을 허용하여</br>최적의 매칭을 찾음
특징 | 각 시점의 값들을 직접 비교 | 시계열 간의 시간적 변동성을</br>고려하여 비교
장점 | 계산이 빠르고 간단함</br>일반적인 시계열 데이터 분석에 적합 | 시간 축의 변동을 허용하여 더 유연한 매칭</br>비슷한 패턴 감지 용이
단점 | 시간의 변동성이나 길이가 다른 시계열 데이터에</br>적용하기 어려움 | 계산 시간이 길고 복잡</br>큰 데이터 셋에는 접하지 않을 수 있음


## 4. K-Means Clustering
### 4.1. 최적의 K 찾기
#### - Elbow Method: Elbow 방법을 사용하여 최적의 K는 5로 정의

<p align= 'center'>
    <figure align= 'center'>
        <img src='./img/inertia.png', title= '인구 연령구분 선행연구'>
        <figcaption>그림3. 인구구조 TimeSeries K-Means Cluster Inertia</figcaption>
    </figure>
</p>

#### - Silhouette Plot: K가 5일 때의 Silhouette 점수를 확인하여 클러스터의 유효성 확인
<p align= 'center'>
    <figure align= 'center'>
        <img src='./img/Silhouette_Plot.png', title= '인구 연령구분 선행연구'>
        <figcaption>그림4. 인구구조 Silhouettte Plot(K=5)</figcaption>
    </figure>
</p>

## 5. 도시유형별 특징

## 최종 결과보고서

** 본 연구는 2023년도 한반도미래인구연구원 연구비 지원에 의한 연구임을 밝힙니다. (KPPIF 23-R04)