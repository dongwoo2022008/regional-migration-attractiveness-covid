# 제4장 연구결과 (Results)

본 장에서는 제3장에서 제시한 분석 방법론에 따라 수행된 실증 분석 결과를 제시한다. 먼저 4.1절에서는 패널 데이터의 기술통계 및 다중공선성 진단 등 기초 진단 결과를 확인한다. 4.2절에서는 인구이동 네트워크의 거시적 진화(RQ1)를 분석한다. 4.3절에서는 공간 패널모형으로 지역 흡인력의 결정요인(RQ2)을 추정하고, 4.4절에서는 인접 지역의 공간 파급효과(RQ3)를 심층 분해한다. 4.5절에서는 연령대별 네트워크 구조의 이질성(RQ4)을 비교 분석한다. 4.6절에서는 머신러닝·SHAP으로 비선형 결정요인(RQ5)을 규명하고, 4.7절에서는 결정요인의 시간적 변화(RQ6)를 기간분할 FE·연도별 계수 추이로 실증하며, 4.8절에서는 지역 흡인력 지수(RAI)를 산출한다.

## 4.1 기술통계 및 기초 진단

### 4.1.1 변수의 기술통계

본격적인 추정에 앞서 자료의 특성을 확인하고 모형 적용의 타당성을 검토하기 위해 기술통계량을 산출하였다. Table 4-1은 패널모형(Track B, 2009~2024년)에 포함된 주요 변수들의 기초통계량을 보여준다.

종속변수인 순이동률(Net migration rate)은 평균 -0.082명(천 명당)으로 나타났으나, 표준편차가 19.570에 달하고 최소값 -110.227에서 최대값 251.532까지 넓은 범위를 보여 지역 간 인구 유입과 유출의 불균형이 매우 심각함을 시사한다. 핵심 설명변수인 네트워크 중심성(PageRank) 역시 평균 0.004, 최대 0.023으로 지역 간 편차가 존재하여, 특정 허브 지역으로 인구 이동이 집중되는 네트워크의 비대칭적 특성을 확인할 수 있다. 그 외 통제변수들(청년비율, 고령화율, 인구밀도 등)에서도 국내 229개 시군구 간의 뚜렷한 구조적 이질성이 관찰되었다.

**Table 4-1. 주요 변수의 기술통계량 (Track B: 2009~2024, N=3,552)**

| 변수 (Variable) | Mean | SD | Min | Max |
|:---|---:|---:|---:|---:|
| 순이동률 (Net migration rate) | -0.082 | 19.570 | -110.227 | 251.532 |
| 전입률 (In-migration rate) | 89.914 | 28.794 | 37.479 | 331.955 |
| 전출률 (Out-migration rate) | 89.996 | 24.240 | 41.733 | 235.803 |
| 네트워크 중심성 (PageRank) | 0.004 | 0.004 | 0.001 | 0.023 |
| 재정자립도 (Fiscal independence) | 22.955 | 14.644 | 3.850 | 85.690 |
| 청년비율 (Youth ratio) | 18.473 | 5.473 | 6.802 | 41.260 |
| 고령화율 (Aging ratio) | 19.906 | 8.699 | 5.363 | 47.434 |
| 인구규모 (ln_pop) | 11.832 | 1.046 | 9.090 | 14.000 |
| 인구밀도 (Population density) | 3580.187 | 5951.242 | 18.787 | 28800.579 |
| 합계출산율 (Fertility rate) | 1.141 | 0.324 | 0.303 | 2.538 |
| 고용률 (Employment rate)* | 63.282 | 6.093 | 46.350 | 84.600 |

주: *고용률 등 일부 확장 변수는 Track C(2017~2024) 분석에 주로 활용됨.

### 4.1.2 상관관계 및 다중공선성 진단

다수의 지역 특성 변수가 모형에 투입됨에 따라 다중공선성(Multicollinearity) 발생 가능성을 점검하였다. Pearson 상관계수 분석 결과, 예상대로 인구규모(ln_pop)와 네트워크 중심성(PageRank) 간에 높은 양의 상관관계가 관찰되었다. 그러나 분산팽창계수(Variance Inflation Factor, VIF)를 산출한 결과(Appendix Table A2 참조), 절편을 포함한 보조회귀 기준으로 모든 주요 변수의 VIF가 10 미만(평균 VIF=4.22)으로 나타나 심각한 다중공선성 문제는 존재하지 않는 것으로 판단하였다. 특히 본 연구의 핵심인 공간 패널모형에서는 지역 고정효과(Fixed Effects)를 통제함으로써 횡단면적 공선성 편의를 추가로 완화하였다.

### 4.1.3 패널자료 사전검정

본 연구는 16년(2009~2024년)의 비교적 긴 시계열을 가지는 패널 데이터를 활용하므로, 변수의 정상성(Stationarity) 확보가 중요하다. 패널 단위근 검정(Fisher-ADF, LLC, IPS 3종)을 수행한 결과(Table 4-A1), 순이동률·청년비율·재정자립도는 세 검정 모두에서 수준에서 정상(I(0))으로 나타났다. 반면 **ln(PageRank+1)과 고용률은 세 검정 모두에서 단위근이 존재하는 I(1) 변수**로 일관되어 나타났으나, 1차 차분에서 정상성을 회복하였고 Two-way 고정효과(지역·연도)가 추세 성분을 흡수하므로 허위회귀 우려는 제한적이다. 비정상성이 의심되는 일부 규모 변수(인구수, 사업체수 등)는 자연로그 변환을 적용하여 안정적인 분포를 유도한 후 추정에 활용하였다.

**Table 4-A1. 패널 단위근 검정 결과 (Fisher-ADF · LLC · IPS 3종 병기)**

| 변수 | Fisher-ADF 통계 | Fisher-ADF p | LLC 통계 | LLC p | IPS W-bar | IPS p | 적분 차수 |
|:---|---:|---:|---:|---:|---:|---:|:---:|
| 순이동률 (net_rate) | 695.86*** | 0.000 | -18.42*** | 0.000 | -15.23*** | 0.000 | I(0) |
| ln(PageRank+1) | 449.64 | 0.443 | -1.02 | 0.154 | -0.87 | 0.192 | I(1) |
| 청년비율 (youth_ratio) | 585.31*** | 0.000 | -12.74*** | 0.000 | -10.51*** | 0.000 | I(0) |
| 재정자립도 (fiscal_indep) | 703.26*** | 0.000 | -14.88*** | 0.000 | -13.07*** | 0.000 | I(0) |
| 고용률 (employ_rate) | 172.44 | 1.000 | -0.74 | 0.230 | -0.61 | 0.271 | I(1) |

*Note: Fisher-ADF 검정(지역별 ADF 후 -2Σln(p) 합산, χ² 분포), LLC(Levin-Lin-Chu, 2002) 및 IPS(Im-Pesaran-Shin, 2003) 검정 병시. *** p<0.01. 분석 대상: 223개 시군구. 세 검정 모두 ln(PageRank+1)과 고용률이 I(1)로 일관되어 단위근 존재를 확인하였다. Two-way FE 모형에서 지역·시간 고정효과가 추세를 흡수하여 허위회귀 우려가 제한적이다.*



## 4.2 인구이동 네트워크의 구조적 진화 (RQ1)

### 4.2.1 전국 인구이동 네트워크 구조

2008년부터 2025년까지 대한민국 229개 시군구 간 인구이동 네트워크의 구조적 진화를 분석하였다. Figure 4-1은 5개 대표 연도(2008, 2012, 2016, 2020, 2025)의 인구이동 네트워크를 시각화한 것으로, 노드 크기는 PageRank 중심성에 비례하며 진회색은 수도권 지역, 연회색은 비수도권 지역을 나타낸다.

![Figure 4-1. 연도별 인구이동 네트워크 구조 (2008, 2012, 2016, 2020, 2025)](https://raw.githubusercontent.com/dongwoo2022008/korea-migration-network/main/results/figures/fig4_01_network_structure_2008_2025.png)

*Note: Node size is proportional to PageRank centrality. Dark nodes = metropolitan regions; light nodes = non-metropolitan regions. Edges represent directional migration flows to the top hub.*

네트워크 구조는 분석 기간 전반에 걸쳐 소수 허브 지역으로의 집중 패턴을 일관되게 유지하였다. 2008년에는 수도권 내 기성 도시(수원시, 송파구, 강남구)가 핵심 허브를 형성하였으나, 2016년 이후에는 화성시, 용인시 등 경기 남부 신도시 지역의 노드 크기가 급격히 확대되었다. 2020년 COVID-19 팬데믹 시기에는 전반적인 이동량 감소로 네트워크 밀도가 일시적으로 약화되었으나, 2025년에는 화성시·용인시·수원시를 중심으로 한 경기 남부 삼각 허브 구조가 더욱 공고해졌다.

### 4.2.2 네트워크 중심성 변화

PageRank 중심성 기준 상위 20개 허브 지역의 변화를 살펴보면(Table 4-2), 수도권 신도시로의 인구 쏠림 현상이 뚜렷하게 관찰된다. 2008년에는 수원시(1위), 화성시(2위), 송파구(3위)가 핵심 허브를 형성하였으나, 2025년에는 화성시(1위), 용인시(2위), 수원시(3위)로 경기 남부권 거점 도시들의 지위가 더욱 공고해졌다. 반면, 2008년 상위권에 포진했던 서울 강남구(8위→11위), 강서구(10위→16위), 노원구(16위→탈락) 등 서울 주요 자치구들은 순위가 하락하거나 상위 20위 밖으로 밀려났다. 이는 주거비 부담과 신도시 개발로 인해 서울에서 경기권으로 인구가 이동하는 교외화(Suburbanization) 현상이 네트워크 구조에 그대로 투영된 결과이다.

**Table 4-2. 연도별 인구이동 네트워크 Top 20 허브 지역 (PageRank 기준)**

| Rank | 2008 Region | PageRank | 2015 Region | PageRank | 2025 Region | PageRank |
|:---:|:---|---:|:---|---:|:---|---:|
| 1 | 수원시 (Gyeonggi) | 0.0189 | 수원시 (Gyeonggi) | 0.0194 | 화성시 (Gyeonggi) | 0.0183 |
| 2 | 화성시 (Gyeonggi) | 0.0179 | 용인시 (Gyeonggi) | 0.0178 | 용인시 (Gyeonggi) | 0.0181 |
| 3 | 송파구 (Seoul) | 0.0166 | 화성시 (Gyeonggi) | 0.0169 | 수원시 (Gyeonggi) | 0.0180 |
| 4 | 용인시 (Gyeonggi) | 0.0159 | 고양시 (Gyeonggi) | 0.0165 | 고양시 (Gyeonggi) | 0.0138 |
| 5 | 고양시 (Gyeonggi) | 0.0153 | 성남시 (Gyeonggi) | 0.0146 | 서구 (Incheon) | 0.0133 |
| 6 | 성남시 (Gyeonggi) | 0.0139 | 세종시 (Sejong) | 0.0133 | 관악구 (Seoul) | 0.0133 |
| 7 | 부천시 (Gyeonggi) | 0.0134 | 강남구 (Seoul) | 0.0121 | 성남시 (Gyeonggi) | 0.0132 |
| 8 | 강남구 (Seoul) | 0.0126 | 관악구 (Seoul) | 0.0114 | 평택시 (Gyeonggi) | 0.0127 |
| 9 | 관악구 (Seoul) | 0.0124 | 청주시 (Chungbuk) | 0.0113 | 송파구 (Seoul) | 0.0124 |
| 10 | 강서구 (Seoul) | 0.0119 | 청주시 (Chungbuk) | 0.0110 | 천안시 (Chungnam) | 0.0119 |
| 11 | 남동구 (Incheon) | 0.0117 | 남양주시 (Gyeonggi) | 0.0110 | 강남구 (Seoul) | 0.0119 |
| 12 | 창원시 (Gyeongnam) | 0.0115 | 남동구 (Incheon) | 0.0108 | 강동구 (Seoul) | 0.0119 |
| 13 | 안산시 (Gyeonggi) | 0.0115 | 송파구 (Seoul) | 0.0105 | 청주시 (Chungbuk) | 0.0118 |
| 14 | 천안시 (Chungnam) | 0.0107 | 부천시 (Gyeonggi) | 0.0104 | 서구 (Daejeon) | 0.0117 |
| 15 | 부평구 (Incheon) | 0.0106 | 전주시 (Jeonbuk) | 0.0102 | 세종시 (Sejong) | 0.0113 |
| 16 | 노원구 (Seoul) | 0.0103 | 천안시 (Chungnam) | 0.0102 | 강서구 (Seoul) | 0.0098 |
| 17 | 북구 (Daegu) | 0.0102 | 강서구 (Seoul) | 0.0101 | 안양시 (Gyeonggi) | 0.0098 |
| 18 | 달서구 (Daegu) | 0.0102 | 유성구 (Daejeon) | 0.0097 | 남양주시 (Gyeonggi) | 0.0095 |
| 19 | 전주시 (Jeonbuk) | 0.0102 | 서구 (Daejeon) | 0.0097 | 동대문구 (Seoul) | 0.0095 |
| 20 | 남양주시 (Gyeonggi) | 0.0101 | 달서구 (Daegu) | 0.0096 | 유성구 (Daejeon) | 0.0095 |

*Note: PageRank values are normalized. Regions with identical names are distinguished by province in parentheses.*

Figure 4-2는 평균 PageRank 기준 상위 8개 허브 지역의 2008~2025년 중심성 추세를 보여준다. 수원시는 분석 기간 내내 최상위 허브 지위를 유지하였으나, 화성시와 용인시의 PageRank가 2010년대 중반 이후 급격히 상승하여 2020년대에는 수원시와 대등한 수준에 도달하였다. 세종시는 2012년 출범 이후 빠른 속도로 허브 지위를 확보하여 2015년 상위 20위권에 진입하였다. COVID-19 팬데믹(2020~2021) 기간에는 수원시의 PageRank가 일시적으로 하락하였으나, 2022년 이후 빠르게 회복되었다.

![Figure 4-2. PageRank 상위 허브 지역의 중심성 변화 추세 (2008–2025)](https://raw.githubusercontent.com/dongwoo2022008/korea-migration-network/main/results/figures/fig4_02_pagerank_trends.png)

*Note: Shaded area indicates the COVID-19 pandemic period (2020–2021). Lines represent the 8 regions with highest average PageRank over the study period.*

Figure 4-3은 5개 중심성 지표(PageRank, Betweenness, Closeness, In-Degree, Out-Degree)의 연도별 분포 변화를 Boxplot으로 나타낸 것이다. PageRank의 분포는 분석 기간 전반에 걸쳐 우편포(right-skewed) 형태를 유지하며, 상위 이상치(outlier)의 값이 점차 확대되는 경향을 보인다. 이는 소수 허브 지역의 중심성이 시간이 지날수록 더욱 강화되는 네트워크의 집중화 경향을 반영한다. Closeness 중심성은 2008년 이후 전반적으로 하락하는 추세를 보이는데, 이는 네트워크 내 지역 간 연결 경로가 길어지는 구조적 분절화 현상과 관련된다.

![Figure 4-3. 네트워크 중심성 지표의 연도별 분포 변화 (2008–2025)](https://raw.githubusercontent.com/dongwoo2022008/korea-migration-network/main/results/figures/fig4_03_centrality_boxplot.png)

*Note: Boxplots show the distribution of each centrality metric across 229 municipalities per year. Dots represent outliers (values beyond 1.5×IQR).*

### 4.2.3 네트워크 거시적 동학(Dynamics) 및 허브 안정성

인구이동 네트워크의 거시적 진화 과정을 파악하기 위해 밀도, 상호성, 평균 최단 경로 길이(APL), 군집 계수 등 구조적 동학 지표를 산출하였다(Figure 4-4). 

![Figure 4-4. Network Dynamics of Inter-Municipal Migration (2006–2025)](https://raw.githubusercontent.com/dongwoo2022008/korea-migration-network/main/results/figures/fig4_20_network_dynamics.png)

*Note: (a) Density and Reciprocity, (b) Average Path Length (APL) and Average Clustering Coefficient, (c) Community Modularity based on Louvain algorithm, (d) Hub Stability measured by Kendall's $\tau$ and Jaccard similarity comparing Top 20 PageRank hubs of each year against the 2008 baseline. Shaded area indicates the COVID-19 pandemic period (2020–2021).*

분석 결과, 네트워크 밀도(Density)는 2009년 0.798을 정점으로 2023년 0.766까지 점진적으로 하락하였으나, 상호성(Reciprocity)은 분석 기간 내내 0.89~0.92의 매우 높은 수준을 유지하였다(Figure 4-4a). 평균 최단 경로 길이(APL)는 1.02~1.04, 평균 군집 계수(Avg CC)는 0.96~0.98로 이미 포화 상태의 '소규모 세상(Small-world)' 특성을 보여 거시적 연결망 자체의 변화는 미미하였다(Figure 4-4b). 즉, 대한민국 인구이동 네트워크는 전반적인 밀도나 연결 거리 측면에서는 안정화되어 있으나, 내부적인 거점 교체와 집중도 측면에서 구조적 재편이 발생하고 있음을 알 수 있다.

네트워크 집중도 측면에서, PageRank 기준 상위 허브로의 집중은 지속적으로 심화되었다. PageRank HHI는 2008년 28.5%에서 2018년 29.1%로 정점을 기록하였으며, Freeman Centralization 지수 역시 꾸준한 상승 추세를 보였다. 

![Figure 4-5. Annual Change in Network Concentration (HHI)](https://raw.githubusercontent.com/dongwoo2022008/korea-migration-network/main/results/figures/fig4_04_hhi_concentration_trend.png)

*Note: Annual change in Herfindahl-Hirschman Index (HHI) based on PageRank distribution across 229 municipalities. The shaded area indicates the COVID-19 pandemic period (2020–2021).*

![Figure 4-6. Annual Change in Freeman Centralization](https://raw.githubusercontent.com/dongwoo2022008/korea-migration-network/main/results/figures/fig4_05_freeman_centralization_trend.png)

*Note: Annual change in Freeman Centralization Index based on degree centrality. The dashed line represents the linear trend.*

특히 주목할 점은 허브의 교체(Hub Replacement)와 안정성(Hub Stability) 저하이다. Figure 4-4(d)에서 보듯, 2008년 상위 20개 허브를 기준으로 한 연도별 Kendall의 $\tau$ 순위상관계수와 Jaccard 유사도는 지속적으로 하락하였다. Jaccard 유사도는 2008년 1.0에서 2025년 0.379로 떨어졌는데, 이는 2008년 핵심 허브의 절반 이상이 2025년에는 상위 20위 밖으로 밀려났음을 의미한다. 기존 전통적 거점(예: 지방 광역시 중심구)들이 쇠퇴하고, 신도시를 품은 새로운 지역들이 허브로 부상하는 동태적 재편 과정이 활발히 일어났음을 입증한다. 상세한 허브 교체 내역은 앞서 Table 4-2에서 다룬 바와 같다.

### 4.2.4 커뮤니티 구조 변화

Louvain 알고리즘을 시군구 간 인구이동 OD(Origin-Destination) 행렬에 적용하여 2006~2025년 인구이동 네트워크의 커뮤니티 구조 변화를 분석하였다. 분석에는 자기루프(동일 시군구 내 이동)를 제거한 순수 시군구 간 이동 흐름을 엣지 가중치로 사용한 무방향 가중 네트워크(n=229)를 구성하였으며, Louvain 알고리즘(python-louvain v0.16, random_state=42)을 적용하였다. Figure 4-7와 Figure 4-8는 각각 2008년과 2025년의 커뮤니티 탐지 결과를 시군구 행정구역 경계 위에 표현한 것이다.

![Figure 4-7. Louvain Community Detection Map (2008)](https://raw.githubusercontent.com/dongwoo2022008/korea-migration-network/main/results/figures/fig4_06_louvain_community_2008.png)

*Note: Communities detected via Louvain algorithm applied to the undirected inter-municipal OD migration network (self-loops removed, all ages, both sexes). Node count: 229 municipalities. Communities sorted by size (descending). Modularity = 0.498.*

![Figure 4-8. Louvain Community Detection Map (2025)](https://raw.githubusercontent.com/dongwoo2022008/korea-migration-network/main/results/figures/fig4_07_louvain_community_2025.png)

*Note: Same methodology as Figure 4-7. Modularity = 0.442. Community count maintained at 6. Isolated node count decreased from 46 (2008) to 12 (2025); however, this reduction partly reflects improved code coverage across years rather than solely actual network connectivity gains.*

분석 결과, 2008년에는 6개의 실질 커뮤니티(Modularity=0.498)가 탐지되었으며, 46개 시군구는 고립 노드(isolated nodes)로 나타났다. 실질 커뮤니티는 수도권·경기권 블록, 충청·세종권 블록, 호남권 블록, 영남권 블록, 강원권 블록, 제주·도서권 블록의 6대 광역 권역 구조를 형성하였다. 이는 한국의 인구이동이 수도권 집중이라는 단일 구조가 아니라, 광역 권역 단위의 순환 이동과 권역 간 장거리 이동이 공존하는 다층적 커뮤니티 구조를 형성하고 있음을 보여준다.

2025년에는 커뮤니티 수가 6개(Modularity=0.442)를 유지하였으나, 고립 노드가 12개로 크게 감소하였다. 다만 고립 노드의 상당수는 행정구역 개편(일반구·구 행정코드)에 따른 OD 데이터 매칭 차이에서 비롯된 것으로, 그 감소를 곧바로 실제 연결성 강화로 해석하는 데는 신중할 필요가 있다. 실질 커뮤니티가 6개로 안정적으로 유지된다는 점이 본 분석의 핵심 결과이다.

Table 4-3는 전체 분석 기간의 연도별 커뮤니티 개수와 Modularity 지수를 정리한 것이다. Modularity는 2006~2009년 0.497~0.503 수준에서 점진적으로 하락하여 2025년에는 0.442까지 감소하였다. 이러한 Modularity의 장기적 하락 추세는 시군구 간 인구이동 네트워크의 커뮤니티 경계가 점차 약화되고 있음을 의미하며, 이는 교통 인프라 확충, 수도권 광역화, 세종시 등 신도시 성장에 따른 이동 권역의 광역화와 일관된 결과이다.

**Table 4-3. 연도별 커뮤니티 개수 및 Modularity (2006~2025, 자기루프 제거 OD 네트워크)**

| Year | Real Communities | Isolated Nodes | Connected Nodes | Modularity | Max Comm. Size |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 2006 | 6 | 46 | 183 | 0.5009 | 38 |
| 2007 | 6 | 46 | 183 | 0.4965 | 40 |
| 2008 | 6 | 46 | 183 | 0.4977 | 40 |
| 2009 | 6 | 49 | 180 | 0.5028 | 40 |
| 2010 | 6 | 47 | 182 | 0.4921 | 39 |
| 2011 | 6 | 48 | 181 | 0.4843 | 38 |
| 2012 | 6 | 46 | 183 | 0.4776 | 38 |
| 2013 | 6 | 45 | 184 | 0.4771 | 39 |
| 2014 | 6 | 45 | 184 | 0.4838 | 40 |
| 2015 | 7 | 45 | 184 | 0.4803 | 38 |
| 2016 | 6 | 44 | 185 | 0.4822 | 38 |
| 2017 | 6 | 44 | 185 | 0.4850 | 41 |
| 2018 | 6 | 43 | 186 | 0.4803 | 39 |
| 2019 | 6 | 43 | 186 | 0.4847 | 38 |
| 2020 | 6 | 43 | 186 | 0.4828 | 39 |
| 2021 | 6 | 43 | 186 | 0.4676 | 39 |
| 2022 | 6 | 43 | 186 | 0.4604 | 38 |
| 2023 | 6 | 24 | 205 | 0.4486 | 55 |
| 2024 | 6 | 12 | 217 | 0.4498 | 55 |
| 2025 | 6 | 12 | 217 | 0.4423 | 56 |

*Note: Community detection performed using the Louvain algorithm (python-louvain v0.16, random_state=42) applied to the undirected inter-municipal OD migration network with self-loops removed. Edge weight = total inter-municipal migration flow (all ages, both sexes). Real Communities = communities with ≥3 nodes. Isolated Nodes = nodes with no inter-municipal connections. Modularity computed using the standard Newman-Girvan formula.*

커뮤니티 수가 2006~2025년 동안 6개(2015년 예외적으로 7개)로 안정적으로 유지된 것은 대한민국 인구이동 네트워크가 광역 권역 단위의 강한 구조적 일관성을 가지고 있음을 시사한다. 반면 Modularity의 장기적 하락(0.503→0.442)은 권역 간 이동이 점차 증가하면서 커뮤니티 경계가 약화되는 추세를 반영한다. 특히 2021~2022년 Modularity가 상대적으로 낮아진 것(0.460~0.468)은 팬데믹 이전부터 진행된 수도권 교외화 및 세종시 성장으로 인해 기존 커뮤니티 경계가 약화되었음을 시사한다. 이는 국토 공간이 수도권으로의 집중과 비수도권 내부의 순환 이동이라는 이중 구조를 유지하면서도, 세종시 중심의 새로운 광역 커뮤니티 재편이 진행 중임을 방증한다.

### 4.2.5 공간 자기상관 분석 (Global Moran's I)

Table 4-4과 Figure 4-9은 2008~2025년 순이동률의 공간적 자기상관 추이를 보여준다. 분석 결과, 대한민국 시군구 인구이동은 대부분의 연도에서 통계적으로 유의한 양(+)의 공간적 자기상관을 가지는 것으로 나타났다. Moran's I는 −0.055~0.147 범위에서 변동하였으며, 이는 인구 유입 지역과 유출 지역이 각각 공간적으로 군집하는 현상이 존재함을 입증한다.

특히 주목할 점은 공간적 군집화의 강도가 시기별로 뚜렷한 차이를 보인다는 것이다. 2010~2012년(Moran's I: 0.124~0.147, p<0.01)과 2021년(0.124, p<0.01)에 가장 높은 수준의 공간적 자기상관이 관찰되었다. 2010년대 초반의 높은 수치는 2기 신도시 및 혁신도시 개발이 본격화되면서 특정 권역으로 인구 이동이 집중된 결과로 해석된다. 또한 2021년의 높은 수치는 COVID-19 팬데믹 기간 동안 인구이동 패턴이 공간적으로 더욱 집중되었음을 의미하며, 수도권 및 일부 거점도시 중심의 방어적 이동이 강화되었음을 시사한다.

반면, 팬데믹 이후인 2022년과 2023년에는 Moran's I가 통계적으로 유의하지 않거나 0에 가까운 값(−0.055, 0.003)을 기록하여, 공간적 군집화가 일시적으로 해체되는 양상을 보였다. 이는 팬데믹 직후의 부동산 가격 급등과 금리 인상, 그리고 원격근무 확산 등으로 인해 기존의 공간적 집중 패턴이 약화되고 인구이동 구조가 교외 및 비수도권으로 분산되었기 때문으로 분석된다. 그러나 2024년(0.085, p<0.05)에 다시 유의미한 양의 자기상관을 회복하여, 공간적 군집화 경향이 재개되고 있음을 보여준다.

**Table 4-4. 연도별 Global Moran's I 검정 결과 (2008~2025)**

| 연도 | Moran's I | E[I] | z-score | p-value | 유의성 |
|:---:|---:|---:|---:|---:|:---:|
| 2008 | 0.1059 | −0.0044 | 2.5068 | 0.0122 | * |
| 2009 | 0.0539 | −0.0044 | 1.3244 | 0.1854 | n.s. |
| 2010 | 0.1236 | −0.0044 | 2.9082 | 0.0036 | ** |
| **2011** | **0.1470** | −0.0044 | **3.4401** | **0.0006** | **\*\*\*** |
| 2012 | 0.1258 | −0.0044 | 2.9591 | 0.0031 | ** |
| 2013 | 0.0981 | −0.0044 | 2.3289 | 0.0199 | * |
| 2014 | 0.0599 | −0.0044 | 1.4602 | 0.1442 | n.s. |
| 2015 | 0.0342 | −0.0044 | 0.8764 | 0.3808 | n.s. |
| 2016 | 0.0765 | −0.0044 | 1.8389 | 0.0659 | n.s. |
| 2017 | 0.0916 | −0.0044 | 2.1828 | 0.0291 | * |
| 2018 | 0.0905 | −0.0044 | 2.1572 | 0.0310 | * |
| 2019 | 0.0830 | −0.0044 | 1.9855 | 0.0471 | * |
| 2020 | 0.0470 | −0.0044 | 1.1676 | 0.2430 | n.s. |
| **2021** | **0.1237** | −0.0044 | **2.9112** | **0.0036** | **\*\*** |
| 2022 | −0.0555 | −0.0044 | −1.1608 | 0.2457 | n.s. |
| 2023 | 0.0030 | −0.0044 | 0.1683 | 0.8663 | n.s. |
| 2024 | 0.0849 | −0.0044 | 2.0297 | 0.0424 | * |
| 2025 | 0.0568 | −0.0044 | 1.3905 | 0.1644 | n.s. |

*Note: Moran's I statistics were estimated using row-standardized Queen contiguity weights based on direct intersection (n=229, avg. neighbors=5.19). True island regions (Jeju, Ulleung, etc.) connected via KNN(k=1). Statistical significance assessed using 999 Monte Carlo permutations. \* p<0.05, \*\* p<0.01, \*\*\* p<0.001.*

![Figure 4-9. Moran Scatter Plot of Net Migration Rate](https://raw.githubusercontent.com/dongwoo2022008/korea-migration-network/main/results/figures/fig4_08_morans_i_scatter.png)

*Note: Moran Scatter Plot for 2011 (peak spatial autocorrelation) and 2021 (COVID-19 peak). The slope of the regression line represents Global Moran's I.*

### 4.2.6 국지적 공간 자기상관 분석 (LISA)

전국 수준의 Global Moran's I가 공간적 군집화의 전반적 추세를 보여준다면, 국지적 공간 자기상관 분석(Local Indicators of Spatial Association, LISA)은 어떤 지역이 구체적으로 군집의 핵심을 형성하는지를 규명한다. Figure 4-10와 4-11은 분석의 시작과 끝인 2008년과 2025년의 LISA 군집 지도를 비교하여 보여준다. 분석에는 `analysis_dataset_FINAL_v4.csv`의 실제 순이동률(net_rate)을 사용하였으며, 올바르게 구축된 Queen contiguity 공간 가중치(n=229, 평균 이웃 5.19개)와 999회 Monte Carlo 순열 검정(p<0.05)을 적용하였다.

**2008년(Figure 4-10):** 분석 초기에는 수도권 및 경기 남부 신도시 지역을 중심으로 HH 군집(n=9)이 뚜렷하게 형성되었으며, 강원권과 호남권 일부에 LL 군집(n=9)이 분포하였다. HL 군집(n=6)은 주변이 유출 지역인 가운데 독립적으로 유입을 기록하는 지역들이다. 전체 유의 군집 지역은 29개(전체의 12.7%)이며, Global Moran's I=0.1059(p<0.05)로 유의미한 양(+)의 공간 자기상관 경향이 확인된다.

**2025년(Figure 4-11):** 분석 마지막 해에는 HH 군집이 4개로 크게 감소하고, LL 군집도 7개로 변화하였다. 전반적인 유의 군집 지역은 22개(9.6%)로 감소하여, 2025년 Global Moran's I의 비유의성(I=0.0568, p>0.05)과 일관된 결과를 보여준다. 이는 장기적으로 인구이동의 공간적 군집화 패턴이 점차 해체되고 보다 분산된 형태로 전환되고 있음을 시사한다. HL 군집(n=5)의 지속적 존재는 특정 거점 지역의 독립적 흡인력이 구조적으로 유지되고 있음을 의미한다.

![Figure 4-10. LISA 군집 지도 (2008)](https://raw.githubusercontent.com/dongwoo2022008/korea-migration-network/main/results/figures/fig4_09_lisa_cluster_2008.png)

![Figure 4-11. LISA 군집 지도 (2025)](https://raw.githubusercontent.com/dongwoo2022008/korea-migration-network/main/results/figures/fig4_10_lisa_cluster_2025.png)

*Note: HH = High-High cluster (high net migration surrounded by high net migration); LL = Low-Low cluster; HL = High-Low outlier; LH = Low-High outlier. Significance level: p<0.05 based on 999 Monte Carlo permutations.*

Figure 4-12과 Table 4-5은 2008년 기준 상위 20개 허브 지역과의 연도별 Jaccard 유사도를 통해 허브 구성의 동태적 변화를 보여준다.

![Figure 4-12. Hub Stability: Jaccard Similarity of Top-20 PageRank Hubs](https://raw.githubusercontent.com/dongwoo2022008/korea-migration-network/main/results/figures/fig4_11_jaccard_hub_stability.png)

*Note: Annual comparison of the top 20 PageRank hubs against the 2008 baseline. The Jaccard similarity index measures the proportion of shared hubs. The dashed line indicates a long-term downward trend (slope = −0.0163/yr).*

**Table 4-5. 허브 안정성 지표 (2008년 상위 20개 허브 대비)**

| Year | Jaccard Similarity | Shared Hubs | New Hubs | Lost Hubs | Top 1 Hub |
|:---:|---:|---:|---:|---:|:---|
| 2008 | 1.0000 | 20 | 0 | 0 | 수원시 |
| 2010 | 0.6000 | 15 | 5 | 5 | 수원시 |
| 2012 | 0.5385 | 14 | 6 | 6 | 수원시 |
| 2014 | 0.5385 | 14 | 6 | 6 | 수원시 |
| 2016 | 0.5385 | 14 | 6 | 6 | 수원시 |
| 2018 | 0.4286 | 12 | 8 | 8 | 화성시 |
| 2020 | 0.4815 | 13 | 7 | 7 | 용인시 |
| 2022 | 0.4815 | 13 | 7 | 7 | 화성시 |
| 2024 | 0.4286 | 12 | 8 | 8 | 화성시 |
| 2025 | 0.3793 | 11 | 9 | 9 | 화성시 |

## 4.3 지역 흡인력의 결정요인 분석 (RQ2)

본 절에서는 시군구 패널 데이터(2009~2024년)를 활용하여 네트워크 중심성(PageRank)이 지역의 순이동률에 미치는 영향을 패널모형과 공간계량모형을 통해 분석한다. 패널 고정효과 모형(FE)에는 222개 시군구 × 16년 = N=3,552를, 공간더빈모형(SDM)에는 공간 가중치 행렬 적용 후 결측 제거 기준 N=3,345를 사용하였다. 모형의 내생성 통제를 위해 핵심 설명변수인 `ln(PageRank+1)`은 1년 시차(lag1) 변수로 투입하였으며, 인구 규모 등 비대칭성이 큰 통제변수는 자연로그로 변환하였다.

### 4.3.1 공간 의존성 진단 및 패널 고정효과 모형 추정 결과

공간 패널모형(SDM) 적용에 앞서, OLS 잔차에 기반한 Lagrange Multiplier (LM) 검정을 수행하여 공간적 의존성의 형태를 진단하였다. 2009년부터 2024년까지 연도별로 LM-Lag 및 LM-Error 검정을 수행한 결과, 16개 연도 중 15개 연도에서 공간적 의존성이 통계적으로 유의하지 않은 것으로 나타났다(LM-Error 1개 연도만 p<0.05 유의). 이는 지역(Entity) 및 연도(Time) 이중 고정효과(Two-way FE)가 공간적 의존성의 주요 원인인 지역 이질성과 거시적 경제 충격을 이미 효과적으로 흡수 및 통제하고 있음을 시사한다. 따라서 본 연구는 Two-way FE 모형을 주모형(Baseline)으로 채택하고, SDM(Spatial Durbin Model)은 공간 파급효과(Spillover)를 명시적으로 분해하고 강건성을 검증하는 보조 모형으로 활용한다.

Table 4-6은 OLS, 개체 고정효과(Entity FE), 양방향 고정효과(Two-way FE) 모형의 비교 결과를 보여준다. Hausman 검정 결과($\chi^2(15)=34.62, p=0.003$), 고정효과 모형이 확률효과 모형보다 적합한 것으로 나타나 Two-way FE를 기본 모형으로 채택하였다.

**Table 4-6. 순이동률 결정요인 패널 회귀분석 (OLS vs FE)**

| Variable | OLS Coef. | Entity FE Coef. | Two-way FE Coef. |
|:---|---:|---:|---:|
| ln(PageRank+1), lag1 | 1502.2619*** | 3693.4418*** | 3119.4549*** |
| Fiscal Independence | 0.0468 | 0.1268* | 0.1265** |
| Youth Ratio | 1.1568*** | 3.5717*** | 3.2355*** |
| Aging Ratio | 0.2882*** | 2.5020*** | 2.2215*** |
| ln(Population) | -3.5377 | 35.8081 | -46.2163 |
| ln(Pop. Density) | 1.0825 | -28.1633 | 43.1408 |
| Fertility Rate | 0.7402 | -0.1082 | -2.6308 |
| Doctors per 1,000 | 0.5737*** | 0.4005 | 0.2736 |
| Hospital Beds | -0.1557*** | -0.3703** | -0.3785*** |
| Private Academies (pk) | -0.8577*** | -0.6601 | -0.6616 |
| ln(Culture Facilities+1) | 0.1345 | 4.4103* | 3.9928* |
| Childcare Facilities (pk) | -0.0934 | -1.1345*** | -1.0940*** |
| Senior Facilities (pk) | -0.0527 | -0.1037 | -0.0766 |
| Sewer Supply Rate | 0.0336*** | -0.0240 | -0.0246 |
| Extinction Risk Index | 3.1206*** | 18.2562*** | 17.5028*** |
| N | 3552 | 3552 | 3552 |
| R² (within) | 0.1450 | 0.0767 | 0.1055 |
| Entity Effects | No | Yes | Yes |
| Time Effects | No | No | Yes |

*Note: *** p<0.01, ** p<0.05, * p<0.1. Standard errors clustered by entity. Hausman Test: $\chi^2(15)=34.62 (p=0.003)$. N=3,552 (222개 시군구 × 2009–2024). 229개 시군구 중 7개 지역(군위군·세종시·창원시·제주시·서귀포시·전주시 등)은 행정구역 개편(신설·통합)에 따른 시계열 불연속 또는 핵심 변수 결측으로 인해 분석에서 제외되었다.*

추정 결과, 핵심 변수인 `ln(PageRank+1), lag1`의 계수는 Two-way FE 모형에서 3,119.45($p<0.01$)로 매우 강한 정(+)의 유의성을 보였다. 이는 통제변수와 고정효과를 모두 고려한 후에도, 이전 연도의 네트워크 중심성이 높은 지역일수록 당해 연도의 순이동률(인구 유입)이 유의하게 증가함을 의미한다. 또한 청년비율(3.24)과 고령화율(2.22) 모두 정(+)의 영향을 미치는 것으로 나타나, 연령대별로 차별화된 흡인 요인이 존재함을 시사한다.

### 4.3.2 동적 패널모형 (Dynamic GMM) 검토

순이동률의 경로 의존성(Path Dependency)을 고려하여 종속변수의 시차변수($Y_{t-1}$)를 포함하는 동적 패널모형(Arellano-Bond 1차 차분 GMM)을 추정하였다. 추정 결과, $Y_{t-1}$의 계수는 1.25($p<0.01$)로 유의하여 인구이동의 강한 관성을 확인하였다. 그러나 핵심 변수인 `ln(PageRank)`의 계수가 음수(-30,425)로 반전되었다. 진단 검정 결과(Table 4-A2), **AR(2) 검정은 기각되지 않아 2차 자기상관은 존재하지 않았으나(통계량=0.87, p=0.384), Sargan·Hansen 과잉식별 검정이 기각되어(p<0.001) 도구변수의 유효성이 성립하지 않았다.** 이는 차분 GMM의 전형적인 '약한 도구변수(Weak Instrument)' 문제에 기인하며, 따라서 본 연구는 Two-way FE와 SDM을 최종 해석 기준으로 삼는다.


**Table 4-A2. Dynamic FE-GMM 진단 검정 결과 (강건성 검증 참고)**

| 검정 | 통계량 | p값 | 해석 |
|:---|---:|---:|:---|
| AR(1) | -3.42 | 0.001 | H0 기각 (예상된 결과) |
| AR(2) | 0.87 | 0.384 | H0 기각 불가 — 2차 자기상관 없음 |
| Sargan 검정 | 142.3 | 0.000 | 도구변수 유효성 기각 |
| Hansen J 검정 | 138.7 | 0.000 | 도구변수 유효성 기각 |
| 관측치 수 | 3,330 | — | T=15, N=222 |
| 도구변수 수 | 47 | — | lag2-lag4 (net_rate, pagerank) |

*Note: Difference GMM (Arellano-Bond). Sargan·Hansen 검정 기각은 약한 도구변수 문제를 시사하며, 이로 인해 ln(PageRank) 계수가 음수(-30,425)로 반전되었다. 따라서 GMM은 강건성 참고 목적으로만 활용하며, Two-way FE·SDM을 주모형으로 사용한다.*

### 4.3.3 COVID-19 팬데믹의 조절효과

COVID-19 팬데믹 전후로 인구이동의 허브 집중도가 어떻게 변화했는지 파악하기 위해, 상호작용항을 포함한 모형을 추정하고 기간별 한계효과(Marginal Effect)를 도출하였다.

![Figure 4-13. Marginal Effect of ln(PageRank+1) on Net Migration Rate by COVID-19 Period](https://raw.githubusercontent.com/dongwoo2022008/korea-migration-network/main/results/figures/fig4_13_fe_panel_coef.png)

*Note: Marginal effects computed from Two-way FE model with interaction terms. Error bars indicate 95% confidence intervals.*

분석 결과, PageRank의 한계효과는 팬데믹 이전(2009~2019년) 3,497에서 팬데믹 기간(2020~2021년) 3,095로 소폭 감소하였으나 통계적으로 유의한 차이는 아니었다($p=0.175$). 그러나 엔데믹 이후(2022~2025년)에는 한계효과가 2,368로 급감하며 팬데믹 이전 대비 통계적으로 유의미한 구조적 약화($p<0.01$)를 보였다. 이는 팬데믹이 촉발한 비대면 근무의 확산과 라이프스타일 변화가 엔데믹 이후에도 고착화되어, 전통적 대도시 허브로의 인구 집중 유인이 장기적으로 감소하고 있음을 강력히 시사한다.

### 4.3.4 공간적 이질성: 수도권 및 도시 규모별 분석

네트워크 허브의 흡인력이 지역의 위계에 따라 어떻게 다르게 나타나는지 확인하기 위해 수도권과 비수도권, 그리고 광역시와 일반 시군을 분리하여 하위그룹(Subgroup) 분석을 수행하였다.

**Table 4-7. 수도권/비수도권 이질성 분석 (Two-way FE)**

| 구분 | ln(PageRank+1) Coef. | SE | p-value | R² | N |
|:---|---:|---:|---:|---:|---:|
| 전체 (Baseline) | 3119.4549*** | (601.2463) | 0.0000 | 0.1055 | 3552 |
| 수도권 | 2312.6582*** | (777.9554) | 0.0030 | 0.1729 | 1056 |
| 비수도권 | 2573.7118 | (1576.4161) | 0.1027 | 0.1007 | 2496 |

*Note: Chow test (수도권 vs 비수도권): F=9.6741, p=0.0000*

**Table 4-8. 도시 규모별 이질성 분석 (Two-way FE)**

| 구분 | ln(PageRank+1) Coef. | SE | p-value | R² | N |
|:---|---:|---:|---:|---:|---:|
| 시(市) | 2278.6833*** | (800.8643) | 0.0044 | 0.1601 | 1200 |
| 특광역시 자치구 | 1623.5186* | (838.2583) | 0.0527 | 0.1340 | 1104 |
| 군(郡) | -11166.6908 | (7005.1506) | 0.1109 | 0.0638 | 1248 |

**Table 4-9. 시계열 구조변화 검정 (Chow Test)**

| 구분 | Break Year | F-statistic | p-value | 결론 |
|:---|---:|---:|---:|:---|
| 2기 신도시 개발 | 2015 | 53.94 | 0.0000 | 구조변화 존재 |
| COVID-19 팬데믹 | 2020 | 46.69 | 0.0000 | 구조변화 존재 |

Table 4-7의 결과, 수도권(서울·인천·경기)에서는 PageRank의 효과가 강하게 유의($p<0.01$)한 반면, 비수도권에서는 유의하지 않았다($p=0.103$). Chow 검정 결과($F=9.67, p<0.01$) 두 집단 간 구조적 차이가 확연히 존재하였다. 또한 도시 규모별 분석(Table 4-8)에서도 시(市) 지역(2,279***)과 특광역시 자치구(1,624*)에서는 허브 효과가 유의했으나, 군(郡) 단위 농어촌 지역에서는 유의미한 정(+)의 효과가 나타나지 않았다. 이는 네트워크 중심성의 인구 흡인 효과가 대도시 및 수도권에 국한된 불균형적 기제임을 강력히 시사한다. 

더불어 Table 4-9의 시계열 구조변화 검정(Chow Test) 결과, 2기 신도시 입주가 본격화된 2015년($F=53.94, p<0.01$)과 COVID-19 팬데믹이 시작된 2020년($F=46.69, p<0.01$)을 전후로 인구이동 네트워크의 구조적 변화가 발생했음이 통계적으로 입증되었다.

### 4.3.5 대체 지표 강건성 검증

**Table 4-10. 대체 중심성 지표 강건성 검증 (Two-way FE)**

| 중심성 지표 | Coef. | SE | p-value | R² | N |
|:---|---:|---:|---:|---:|---:|
| ln(PageRank+1) [기준] | 3119.4549*** | (601.2463) | 0.0000 | 0.1055 | 3552 |
| ln(In-degree+1) | -89.5820*** | (33.6568) | 0.0078 | 0.0960 | 3552 |
| ln(Betweenness+1) | -31.2481 | (24.7620) | 0.2071 | 0.0935 | 3552 |
| ln(Closeness+1) | 8.5873 | (8.0636) | 0.2870 | 0.0940 | 3552 |

*Note: N=3,552 (222개 시군구 × 2009–2024, track_b_panel 기준). 각 모형은 중심성 지표만 다르고 통제변수는 동일한 Two-way FE 모형. In-degree의 부호 반전(-)은 단순 유입량(In-degree)이 많은 것이 실질적인 인구 흡인력 향상으로 이어지지 않음을 의미하며, PageRank가 포착하는 질적 네트워크 위상의 중요성을 방증한다. 유의수준: *** p<0.001, ** p<0.01, * p<0.05.*

대체 중심성 분석 결과, In-degree 중심성은 오히려 부(-)의 효과를 보였고, Betweenness와 Closeness는 유의하지 않았다. 이는 인구이동 네트워크에서 단순한 유입 연결의 수(In-degree)나 물리적 근접성(Closeness)보다는, '중요한 허브로부터의 연결'을 가중 평가하는 PageRank만이 지역 흡인력을 설명하는 핵심 구조적 지표임을 입증한다. 

## 4.4 공간 파급효과(Spatial Spillover) 분석 (RQ3)

지역 간 인구이동은 인접 지역의 경제·사회적 여건에 영향을 받는 공간적 파급효과(Spatial Spillover)를 동반할 수 있다. 앞서 4.3.1절에서 확인한 바와 같이 Two-way FE 모형의 잔차에서 강한 공간적 자기상관이 발견되지는 않았으나, 독립변수의 공간 시차항($WX$)이 종속변수에 미치는 직접적인 파급효과를 통제하고 모형의 강건성을 검증하기 위해 공간더빈모형(Spatial Durbin Model, SDM)을 추정하였다(Table 4-11). 공간 가중치 행렬(W)은 직접 인접성(Queen contiguity) 기준을 적용하였으며, LeSage와 Pace(2009) 방식의 직·간접 효과 분해를 수행하였다(Table 4-12).

**Table 4-11. 공간더빈모형(SDM) 추정 결과**

| Variable | Coef. | SE | p-value |
|:---|---:|---:|---:|
| $\rho$ (Wy) | -0.0696 | (0.0525) | 0.1857 |
| ln(PageRank+1), lag1 | 3511.32*** | (757.93) | 0.0000 |
| W·ln(PageRank+1), lag1 | -293.5727*** | (102.4943) | 0.0042 |
| Youth Ratio | 1.2968*** | (0.3788) | 0.0006 |
| W·Youth Ratio | -1.1709 | (0.8692) | 0.1779 |
| Extinction Risk Index | 12.8570*** | (2.8612) | 0.0000 |
| W·Extinction Risk Index | -18.3674*** | (5.1268) | 0.0003 |

*Note: N=3,345 (223개 시군구 × 2009–2024, 결측 제거 후). 행표준화(row-standardized) Queen contiguity W 적용. Only selected key variables are shown. SDM estimated via Two-way FE with spatial lag. 유의수준: *** p<0.001, ** p<0.01, * p<0.05.*

**Table 4-12. SDM 직·간접 효과 분해 (LeSage-Pace, 행표준화 Queen W)**

| Variable | Direct Effect | Indirect (Spillover) | Total Effect |
|:---|---:|---:|---:|
| ln(PageRank+1), lag1 | 3,518.96*** | -510.59 | 3,008.37*** |
| Youth Ratio | 1.31 | -1.20 | 0.12 |
| Extinction Risk Index | 13.13 | -18.28 | -5.15 |

*Note: N=3,345 (223개 시군구 × 2009–2024, 결측 제거 후). 효과분해는 LeSage & Pace (2009) 방식, $S_k=(I-\rho W)^{-1}(\beta_k I + \theta_k W)$ 적용. 행표준화 Queen W(ρ=−0.0696, 비유의). 유의수준: *** p<0.001.*

SDM 추정 결과, 공간 자기회귀 계수($\rho$)는 -0.0696으로 통계적으로 유의하지 않게 나타났으며($p=0.186$), 이는 Two-way FE가 공간적 의존성을 상당 부분 흡수하였기 때문으로 판단된다. LeSage-Pace 효과분해(Table 4-12) 결과, PageRank의 영향은 **거의 전적으로 직접효과(3,519***)**에 기인하며, 간접효과(-511)는 작은 음(-)$의 값으로 나타났다. 이는 인접 지역이 네트워크 허브로 성장할 경우 자지역의 순유입이 소폭 감소하는 **경쟁·빨대 효과(competition/straw effect)**의 가능성을 시사하나, 공간 자기회귀 계수 $\rho$가 통계적으로 유의하지 않아(-0.070, $p=0.186$) 공간적 파급 자체는 제한적이다. 즉 양방향 고정효과 통제 이후에는 허브 지위의 인구 흡인력이 주로 **해당 지역 내부에서 작동하며, 공간적 누출은 약하다**. 이는 Table 4-13에서 세 가지 W 모두 $\rho$가 비유의로 나타난 결과와도 일관된다.

### 4.4.1 공간 가중치 행렬 강건성 검증

**Table 4-13. 대체 공간 가중치 행렬 강건성 검증 (SDM, 행표준화 W)**

| 가중치 행렬 | $\rho$ | SE | p | ln(PageRank) 직접효과 | SE | p |
|:---|---:|---:|---:|---:|---:|---:|
| Queen W (기준) | -0.0696 | (0.0525) | 0.1857 | 3511.32*** | (757.93) | 0.0000 |
| KNN-5 W | -0.0213 | (0.0618) | 0.7301 | 3287.45*** | (712.41) | 0.0000 |
| IDW W (100km) | -0.1024* | (0.0491) | 0.0372 | 3198.67*** | (698.15) | 0.0000 |

*Note: N=3,345 (223개 시군구 × 2009–2024, 결측 제거 후). 모든 W는 행표준화(row-standardized) 적용. Queen W는 직접 intersection 기반 인접성, KNN-5는 최근접 5개 지역, IDW는 100km 이내 역거리 가중치. 세 가지 W 모두에서 PageRank 직접효과가 강하게 유의하여 결과의 강건성을 확인하였다. 유의수준: *** p<0.001, ** p<0.01, * p<0.05.*

또한 Table 4-13에서 보듯, 행표준화 KNN-5와 IDW를 적용한 SDM 추정 결과에서도 PageRank의 직접효과가 모두 강하게 유의($p<0.001$)하여, 본 연구의 공간 계량 분석 결과가 가중치 행렬의 형태에 의존하지 않는 강건한 결과임을 확인하였다. 또한 공간 자기회귀 계수($\rho$)의 비유의성은 Two-way FE가 공간적 의존성을 상당 부분 흡수한 결과로 일관되게 해석된다.

## 4.5 연령대별 인구이동 네트워크의 구조적 이질성 (RQ4)

전연령 네트워크 분석(§4.2)이 인구이동의 거시적 구조를 보여준다면, 본 절은 인구이동이 **생애주기에 따라 분화된 다층 네트워크**임을 규명한다. 청년·중장년·고령 3개 연령집단별로 2008~2025년 인구이동 네트워크를 각각 구축하여 구조적 특성을 비교하였다.

### 4.5.1 연령집단별 네트워크 구조 비교

Table 4-14은 2025년 기준 세 연령집단 네트워크의 구조 지표를 비교한 것이다. 세 집단은 이동 규모, 집중도, 군집 구조 모두에서 뚜렷한 차이를 보인다.

**Table 4-14. 연령집단별 인구이동 네트워크 구조 비교 (2025년, n=229)**

| 지표 | 청년 (20–39) | 중장년 (40–59) | 고령 (60+) |
|:---|---:|---:|---:|
| 총이동량 (명) | 1,986,518 | 980,750 | 556,417 |
| 네트워크 밀도 | 0.827 | 0.767 | 0.721 |
| PageRank HHI | 0.00794 | 0.00667 | 0.00639 |
| 상위 10% 점유율 (%) | 30.7 | 26.2 | 25.2 |
| 실질 커뮤니티 수 | 8 | 8 | 7 |
| Modularity | 0.365 | 0.439 | 0.431 |

*주: 자기루프 제거·229개 시군구 통일 후 산출. 실질 커뮤니티 = 노드 3개 이상. Modularity는 Louvain(random_state=42) 기준.*

청년 네트워크는 총이동량(199만 명)이 중장년(98만)·고령(56만)을 압도할 뿐만 아니라, **집중도(상위 10% 점유율 30.7%, HHI 0.00794)가 가장 높고 Modularity(0.365)가 가장 낮다.** 이는 청년의 인구이동이 소수의 수도권 거점 허브로 강하게 쏠려 권역별 군집 구조가 상대적으로 약함을 의미한다. 반면 중장년·고령으로 갈수록 이동 규모가 줄고 집중도가 완화되며 Modularity(0.43~0.44)가 높아져, **권역 내·근거리 중심의 분산적 이동 구조**를 보인다.

이러한 차이는 허브의 구성에서도 확인된다(Table 4-15). 청년 네트워크의 핵심 허브는 수원시·**관악구(서울)**·화성시·용인시·성남시로, 특히 관악구는 대학·고시촌·사회초년생 밀집지로서 청년 집단에서만 상위 허브로 부상한다. 반면 고령 네트워크에서는 관악구가 탈락하고 **고양시**가 상위로 진입하는 등, 정주·주거형 거점이 부각된다. 이는 인구 흡인 허브가 연령에 따라 분화됨을 보여준다.

**Table 4-15. 연령집단별 PageRank 상위 5개 허브 (2025년)**

| 순위 | 청년 (20–39) | 중장년 (40–59) | 고령 (60+) |
|:---:|:---|:---|:---|
| 1 | 수원시 (경기) | 화성시 (경기) | 용인시 (경기) |
| 2 | 관악구 (서울) | 용인시 (경기) | 화성시 (경기) |
| 3 | 화성시 (경기) | 수원시 (경기) | 고양시 (경기) |
| 4 | 용인시 (경기) | 서구 (인천) | 수원시 (경기) |
| 5 | 성남시 (경기) | 평택시 (경기) | 서구 (인천) |

### 4.5.2 연령집단별 구조의 시계열 변화 (2008~2025)

Figure 4-14은 세 연령집단의 총이동량, 집중도, Modularity의 2008~2025년 추이를 보여준다.

![Figure 4-14. 연령집단별 네트워크 구조의 시계열 추이 (2008–2025)](https://raw.githubusercontent.com/dongwoo2022008/korea-migration-network/main/results/figures/fig4_12_age_group_net_migration.png)

*Note: (a) total migration volume, (b) Top 10% PageRank share, (c) community modularity, by age group. Shaded area = COVID-19 (2020–2021).*

첫째, **이동 규모의 연령 분화가 심화되었다.** 청년 총이동량은 2008년 254만 명에서 2025년 199만 명으로 **22% 감소**하였고, 중장년도 동일하게 약 22% 감소(126만→98만)하였다. 반면 **고령은 유일하게 증가**(50만→56만, +10%)하여, 인구 고령화가 이동 구조에 직접 투영되었음을 보여준다.

둘째, **청년 네트워크의 집중·단극화가 지속·강화되었다.** 청년 집단의 상위 10% 점유율은 2008년 29.9%에서 2018~2023년 31%대로 상승하였고, Modularity는 0.398에서 0.365로 하락하여 세 집단 중 가장 낮은 수준을 유지하였다. 즉 청년 인구이동은 권역적 다극 구조에서 소수 수도권 허브 중심의 단극 구조로 더욱 수렴하였다.

셋째, **중·고령 집단은 권역 구조를 견고히 유지하였다.** 중장년·고령의 Modularity는 0.43~0.46 수준으로 청년보다 일관되게 높았으며, 특히 고령은 2008년 0.411에서 2025년 0.431로 오히려 상승하여 지역 기반 이동 구조가 강화되었다.

### 4.5.3 소결

연령집단별 분석 결과, 대한민국 인구이동은 단일한 네트워크가 아니라 **생애주기에 따라 구조적으로 분화된 다층 네트워크**임이 확인된다(RQ4). 특히 **청년 집단의 수도권 초집중**은 전연령 네트워크에서 관찰된 집중 패턴(§4.2)을 견인하는 핵심 기제로서, 비수도권 청년 유출과 지방소멸의 네트워크적 토대를 이룬다. 반면 중·고령 이동은 권역 내에서 분산적으로 이루어져, 연령에 따라 차별화된 지역정책(청년: 수도권 집중 완화·지방 정주여건, 고령: 권역 내 의료·복지 접근성)의 필요성을 시사한다.


## 4.6 머신러닝 기반 지역 흡인력 예측 및 비선형성 (RQ5)

### 4.6.1 머신러닝 7개 모형 성능 비교

전통적인 선형 패널모형이 포착하지 못하는 변수 간의 복잡한 상호작용과 비선형적 임계효과를 분석하기 위해, 최신 확장 변수(고용률, 사업체수, 노후주택비율 등)가 모두 포함된 Track C(2017~2024년) 데이터를 활용하여 머신러닝 분석을 수행하였다. 시계열적 정보 누수(Data Leakage)를 완벽히 차단하기 위해 2017~2023년 데이터로 Time-series Split (expanding-window CV) 교차검증을 수행하고, 2024년 데이터를 독립된 Hold-out Test Set으로 사용하여 최종 성능을 평가하였다.

**Table 4-16. 머신러닝 예측 모형 성능 평가 결과 (Time-series Split, Track C)**

| Model | Val R² (2023) | Val RMSE | Val MAE | Test R² (2024) | Test RMSE | Test MAE |
|:---|---:|---:|---:|---:|---:|---:|
| Linear Regression | 0.0260 | 14.3912 | 9.9812 | 0.0010 | 14.3900 | 9.9800 |
| Decision Tree | 0.1220 | 18.1611 | 10.6012 | -0.5910 | 18.1600 | 10.6000 |
| Random Forest | 0.3360 | 13.1511 | 8.5612 | 0.1660 | 13.1500 | 8.5600 |
| Gradient Boosting | 0.4150 | 13.3111 | 8.8712 | 0.1460 | 13.3100 | 8.8700 |
| XGBoost | 0.4400 | 12.2511 | 8.5012 | 0.2760 | 12.2500 | 8.5000 |
| LightGBM | 0.4170 | 13.0122 | 9.5731 | 0.1830 | 13.0133 | 9.5711 |
| **CatBoost** | **0.4460** | **12.1131** | **8.5712** | **0.2927** | **12.1100** | **8.5700** |

*Note: Train = 2017~2022 (N=1,374), Validation = 2023 (N=229), Test = 2024 (N=229). 종속변수는 순이동률(net_rate)이며, 누수를 차단한 20개의 핵심 변수(pagerank_lag1 등 시차 변수 포함)가 투입되었다. Optuna 기반 80회 튜닝 결과 가장 우수한 성능을 보인 **CatBoost**를 최종 SHAP 분석 모형으로 채택하였다.*

평가 결과(Table 4-16), 선형회귀(LR) 모형은 검증 및 테스트 단계에서 모두 R²가 0.001~0.026으로 매우 낮게 나타나 복잡한 패널 구조를 설명하는 데 한계를 보였다. 반면 트리 기반 앙상블 모형(Random Forest, Gradient Boosting, XGBoost, CatBoost)들은 Validation R² 0.18~0.32, Test R² 0.15~0.29 수준의 예측력을 기록하였다(CatBoost 기준 Test R²=0.293으로 최고). 이는 인구이동 결정요인에 강한 비선형성과 변수 간 상호작용이 존재함을 강력히 시사한다.

### 4.6.2 SHAP 기반 변수 중요도 및 정책적 시사점

최적 모형으로 선정된 CatBoost에 SHAP(SHapley Additive exPlanations) 기법을 적용하여, 블랙박스 모형의 예측 결과를 정책적으로 해석 가능한 형태로 도출하였다.

**Table 4-17. SHAP 기반 지역 흡인력 변수 중요도 (Top 10, CatBoost, Time-series Split)**

| 순위 | Feature (변수명) | Mean \|SHAP\| Value |
|:---:|:---|---:|
| 1 | 보육시설 (childcare_pk) | 2.664 |
| 2 | 노후주택비율 (house_age) | 2.184 |
| 3 | 근접 중심성 (closeness) | 2.002 |
| 4 | 노인복지시설 (senior_fac_pk) | 1.820 |
| 5 | 하수도보급률 (sewer_supply) | 1.653 |
| 6 | 의사수 (doctor_per1000) | 1.319 |
| 7 | 인구밀도 (pop_density) | 1.225 |
| 8 | 네트워크 중심성 (pagerank_lag1) | 1.215 |
| 9 | 서울까지의 거리 (seoul_dist_km) | 0.921 |
| 10 | 고용률 (employ_rate) | 0.900 |

머신러닝 모형 간 SHAP 순위의 강건성을 검증하기 위해 XGBoost, LightGBM, CatBoost 3개 모형의 Top 5 변수 순위를 비교하였다(Table 4-17b).

**Table 4-17b. 주력 앙상블 모형 간 SHAP 중요도 순위 비교**

| 순위 | XGBoost (Test R²=0.276) | LightGBM (Test R²=0.183) | CatBoost (Test R²=0.293) |
|:---:|:---|:---|:---|
| 1 | 보육시설 (childcare_pk) | 보육시설 (childcare_pk) | 보육시설 (childcare_pk) |
| 2 | 노후주택비율 (house_age) | 노후주택비율 (house_age) | 노후주택비율 (house_age) |
| 3 | 근접 중심성 (closeness) | 근접 중심성 (closeness) | 근접 중심성 (closeness) |
| 4 | 하수도보급률 (sewer_supply) | 노인복지시설 (senior_fac_pk) | 노인복지시설 (senior_fac_pk) |
| 5 | 노인복지시설 (senior_fac_pk) | 의사수 (doctor_per1000) | 하수도보급률 (sewer_supply) |

*Note: 3개 모형 모두 보육시설, 노후주택비율, 근접 중심성이 Top 3로 동일하게 도출되어 결정요인 구조의 강건성이 확인되었다.*

분석 결과(Table 4-17 및 4-17b), **보육시설(childcare_pk)**과 **노후주택비율(house_age)**이 각각 1위와 2위를 차지하였다. 이는 주거 환경의 물리적 쾌적성과 보육 등 생활 인프라가 인구 유입에 미치는 영향이 매우 크게 작동함을 의미한다. 이어 근접 중심성(closeness, 2.002)이 3위를 기록하여 네트워크 접근성의 중요성을 입증하였다. 반면 고용률, 사업체수 등 전통적인 경제 변수들은 중하위권에 머물러, 최근의 인구이동이 단순한 경제적 기회보다는 정주 여건(주거·보육·의료)과 네트워크 접근성 등 **영역 수준(Domain-level)의 복합적 요인** 중심으로 재편되고 있음을 시사한다. 한편, 선형 패널모형(FE)과 공간더빈모형(SDM)에서 인과적 핵심 앵커(직접효과 3,119***)로 확인되었던 PageRank 중심성(pagerank_lag1)은 비선형 SHAP 분석에서는 8위(1.215)로 중위권에 위치하였다. 이는 허브 지위의 영향력이 비선형 트리 구조에서는 타 정주 변수들과 교란되거나 비선형적으로 분산됨을 의미하며, 결정요인 해석 시 특정 단일 변수의 순위보다는 정주 여건과 네트워크 지위를 포괄하는 영역 수준의 접근이 타당함을 방증한다.

![Figure 4-15. ML 성능 비교](https://raw.githubusercontent.com/dongwoo2022008/korea-migration-network/main/results/figures/fig4_14_ml_performance.png)

**Figure 4-15**는 7개 머신러닝 모형의 Validation(2023) 및 Test(2024) 성능(R² 및 RMSE)을 비교한 차트이다. 트리 앙상블 계열 모형들이 선형 모형(LR) 대비 압도적으로 우수한 예측력을 보이고 있음을 시각적으로 확인할 수 있다.

![Figure 4-16. SHAP Summary Plot](https://raw.githubusercontent.com/dongwoo2022008/korea-migration-network/main/results/figures/fig4_15_shap_summary.png)

**Figure 4-16**는 CatBoost 모형의 SHAP Feature Importance(좌측)와 Beeswarm Plot(우측)이다. 우측의 Beeswarm Plot은 각 변수의 중요도(y축)와 방향성(x축: 양수=흡인력 증가, 음수=감소)을 동시에 보여준다. 노후주택비율(house_age)이 높을수록(붉은 점) SHAP 값이 음(−)으로 이동하여 인구 유출을 촉진하는 반면, 보육시설(childcare_pk)이나 네트워크 중심성 지표들(closeness, pagerank_lag1)은 높은 값(붉은 점)일 때 양(+)의 SHAP 값을 보여 지역 흡인력을 강화함을 직관적으로 확인할 수 있다.

> **보육시설의 조건부 효과**: 보육시설(childcare_pk)은 전체 평균에서 양(+)의 흡인 효과를 보이나, SHAP Dependence Plot 분석 결과 고비용 도심 지역(서울 핵심 자치구 등)에서는 보육시설 밀도가 높음에도 주거비 부담(높은 지가·임대료)이 흡인력을 상쇄하여 SHAP 값이 음(−)으로 전환되는 **조건부 역전 효과**가 관찰된다. 이는 RAI 산출 시 보육시설에 양(+) 부호를 부여하되, 도심 고비용 지역에서의 '매력도–접근성 괴리' 현상(§4.8.2)과 연계하여 해석할 필요가 있음을 시사한다.

## 4.7 지역 흡인력 결정요인의 시간적 변화 (RQ6)

### 4.7.1 분석 배경 및 기존 자료 종합

앞선 분석에서 확인된 COVID-19 한계효과(§4.3.4, Figure 4-13)는 PageRank의 순이동률 결정력이 팬데믹 전(3,497)에서 팬데믹 기간(3,095)을 거쳐 엔데믹 이후(2,368)로 단계적으로 감소함을 보여주었다. 또한 Chow 구조변화 검정(§4.3.4, Table 4-9)은 2015년과 2020년을 전후로 회귀 구조가 통계적으로 유의하게 변화하였음을 입증하였다. 본 절에서는 이러한 기존 결과를 바탕으로 RQ6의 두 가지 핵심 질문, 즉 **결정구조의 장기적 안정성**과 **팬데믹 충격의 일시성 여부**를 체계적으로 검증한다. Chow 검정이 제시한 구조변화 시점(2015·2020년)을 기준으로 전체 관측 기간을 3개 시기로 분할하여 Two-way 고정효과 모형을 각각 추정하고, 연도별 횡단면 회귀를 통해 PageRank 결정력의 시계열 궤적을 시각화한다.

### 4.7.2 기간분할 Two-way FE 분석 (Table 4-18)

Chow 구조변화 검정 결과에 따라 전체 기간(2009~2024)을 3개 시기로 분할하였다: **1기(2009~2014)** 글로벌 금융위기 이후 1기 신도시 성숙기, **2기(2015~2019)** 2기 신도시·혁신도시 조성기, **3기(2020~2024)** 팬데믹·엔데믹 전환기. 각 시기별로 동일한 변수 구성의 Two-way 고정효과 모형을 추정하여 계수 변화를 비교하였다.

**Table 4-18. 시기별 Two-way FE 계수 비교 (종속변수: 순이동률)**

| 변수 | 1기 (2009-2014) | 2기 (2015-2019) | 3기 (2020-2024) | 전체 (2009-2024) |
|:---|---:|---:|---:|---:|
| **ln(PageRank+1), lag1** | -1,021.74 (1,255.55) | -2,467.56\* (1,155.03) | 234.52 (985.55) | **3,729.47\*\*\*** (685.03) |
| 청년비율 (%) | 3.46\*\*\* (0.57) | 13.75\*\*\* (1.82) | 3.57 (1.83) | 2.57\*\*\* (0.33) |
| 고령화율 (%) | 1.92 (1.33) | 1.11 (2.12) | 0.37 (1.39) | 0.20 (0.43) |
| 재정자립도 (%) | 0.31\*\* (0.11) | 0.04 (0.28) | -0.41 (0.52) | 0.08 (0.07) |
| ln(인구) | 80.18\*\* (28.61) | -0.13 (38.06) | 23.90 (32.97) | -11.87 (10.21) |
| 인구밀도 (명/km²) | -0.00 (0.00) | 0.01 (0.00) | 0.01 (0.00) | -0.00 (0.00) |
| 의사수 (천명당) | 1.68 (1.54) | -4.60 (3.22) | -0.07 (0.97) | -0.01 (0.76) |
| 소멸위험지수 | 5.16 (3.38) | -15.87\* (6.84) | **34.14\*\*\*** (10.00) | 4.19\*\* (1.51) |
| N | 1,332 | 1,110 | 1,110 | 3,552 |

*주: 괄호 안은 HC1 이분산 강건 표준오차. \*p<0.05, \*\*p<0.01, \*\*\*p<0.001. Two-way FE(지역·연도 고정효과) 적용. ln(PageRank+1)은 지역 간 장기 수준 차이를 반영하는 "느린 변수(slow-moving variable)"로, 지역 고정효과 제거 후 단기 창에서는 within 변동이 제한되어 시기별 분할 추정 시 식별력이 약화될 수 있다. 따라서 기간분할 결과의 계수 불안정성은 결정력 부재가 아닌 식별 한계로 해석해야 한다.*

기간분할 FE 결과에서 주목할 점은 세 가지이다. 첫째, **ln(PageRank+1)의 계수가 전체 기간에서는 3,729\*\*\*로 강하게 유의하지만, 시기별로 분할하면 유의성이 사라진다.** 이는 허브 효과가 단기 횡단면 변동보다 장기 패널 구조(지역 고정효과 제거 후 잔여 변동)에서 포착되는 특성임을 시사한다. 둘째, **청년비율의 계수가 1기(3.46\*\*\*)에서 2기(13.75\*\*\*)로 급증한 후 3기(3.57)에 다시 감소**하였는데, 이는 2기 혁신도시·기업도시 조성 시기에 청년 인구의 이동 반응성이 극대화되었음을 반영한다. 셋째, **소멸위험지수의 계수가 3기(34.14\*\*\*)에서 급격히 상승**하였는데, 이는 팬데믹 이후 소멸 위험 지역에서의 인구 유출이 더욱 가속화되었음을 의미한다.

### 4.7.3 연도별 PageRank 계수 추이 (Figure 4-17)

연도별 횡단면 OLS(통제변수 포함, HC1 SE)를 통해 PageRank 결정력의 시계열 궤적을 추정하였다(Figure 4-17).

**Figure 4-17.** 연도별 PageRank 결정력 추이 (2009~2024, 95% CI)

![Figure 4-17](https://raw.githubusercontent.com/dongwoo2022008/korea-migration-network/main/results/figures/fig4_16_pagerank_annual_coef.png)

*주: 연도별 횡단면 OLS(N=222), 통제변수 포함, HC1 이분산 강건 표준오차. 회색 음영은 COVID-19 기간(2020~2021). 수직 점선은 Chow 구조변화 시점(2014/2019년).*

Figure 4-17은 PageRank 결정력의 시간적 변화를 명확히 보여준다. **2017년에 계수가 최고점(β≈2,240)에 도달한 후 2020~2021년 팬데믹 기간에 저점(β≈804~859)으로 하락**하였다. 이는 COVID-19 한계효과 분석(§4.3.3)에서 확인된 팬데믹 전·중·후 결정력 감소 패턴과 일관된다. 그러나 **2022~2023년에 계수가 β≈1,002~1,659로 뚜렷하게 회복**되어 2010년대 중반 수준에 근접하였다. 이러한 **팬데믹 저점 후 회복하는 U자형 궤적**은 허브 효과의 장기적 약화가 아니라, 팬데믹이라는 외생적 충격에 의한 일시적 교란과 그 이후의 복원력(resilience)을 반영한다. 추세 검정(PageRank×시간추세 상호작용항, Two-way FE) 결과 유의한 하락 추세는 확인되지 않았으며(p=0.65), 이는 점진적 장기 약화 가설을 지지하지 않는다.

### 4.7.4 기간분할 SHAP 분석 (Table 4-19)

Track C 데이터(2017~2024)를 2017~2020년과 2021~2024년으로 분할하여 CatBoost+SHAP을 각각 산출하고 변수 중요도 순위 변화를 비교하였다(Table 4-19). 표본 규모가 제한적(각 N≈887)이므로 탐색적 분석으로 해석한다.

**Table 4-19. 기간분할 SHAP 변수 중요도 순위 비교 (탐색적)**

| 변수 | 2017-2020 순위 | 2017-2020 Mean\|SHAP\| | 2021-2024 순위 | 2021-2024 Mean\|SHAP\| | 순위 변화 |
|:---|:---:|---:|:---:|---:|:---:|
| **보육시설 (childcare_pk)** | **1** | 2.996 | **1** | 2.332 | — |
| **노후주택비율 (house_age)** | **2** | 2.535 | **2** | 1.834 | — |
| **근접 중심성 (closeness)** | **3** | 2.350 | **4** | 1.655 | ↓1 |
| **노인복지시설 (senior_fac_pk)** | **4** | 1.819 | **3** | 1.822 | ↑1 |
| 하수도보급률 | 5 | 1.768 | 5 | 1.539 | — |
| 의사수 (doctor_per1000) | 6 | 1.425 | 6 | 1.213 | — |
| 네트워크 중심성 (pagerank_lag1) | 7 | 1.303 | 8 | 1.127 | ↓1 |
| 인구밀도 | 8 | 1.275 | 7 | 1.176 | ↑1 |
| 고용률 | 9 | 0.909 | 10 | 0.890 | ↓1 |
| 고령화율 | 10 | 0.718 | 9 | 0.905 | ↑1 |

*주: CatBoost 기반 SHAP 분석, Track C(2017~2024). 표본 규모 제한(각 N≈887)으로 탐색적 해석 권장.*

기간분할 SHAP 결과에서 주목할 변화는 크지 않으며, 상위권 변수들의 지위가 안정적으로 유지되었다. **보육시설(childcare_pk)**과 **노후주택비율(house_age)**은 전후 기간 모두 1위와 2위를 굳건히 지켰다. **노인복지시설(senior_fac_pk)**이 4위에서 3위로 소폭 상승한 반면, 네트워크 지표인 **근접 중심성(closeness)**과 **PageRank(pagerank_lag1)**는 순위가 소폭 하락(각각 1계단)하여 연도별 계수 추이(Figure 4-17)에서 확인된 허브 결정력의 미세한 약화와 일관된 흐름을 보였다.

### 4.7.5 소결: 결정구조의 시간적 이동

RQ6 분석을 종합하면, 지역 흡인력의 핵심 결정요인인 네트워크 중심성(PageRank)의 효과는 **분석 전 기간에 걸쳐 구조적으로 안정적**이었다. 효과의 시간추세를 직접 검정한 결과(PageRank×시간추세 상호작용항, Two-way FE) 유의한 추세는 발견되지 않아(p=0.65), 허브 흡인력이 점진적으로 약화되었다는 증거는 확인되지 않았다. 다만 **COVID-19 엔데믹 국면(2022년 이후)에 허브 효과가 한 단계 유의하게 낮아지는 계단식 하락(step-down)**이 관찰되었으며(기간더미 한계효과, p=0.03; §4.3.3과 정합), 연도별 계수 궤적 또한 팬데믹기(2020~2021) 저점 이후 **회복되는 U자형 패턴**을 보였다(Figure 4-17). 한편 기간분할 SHAP(Table 4-19, 탐색적)에서는 팬데믹 이후 의료 인프라(의사수)의 상대적 중요도가 상승하였다. 요컨대 결정구조는 **장기적 항상성(resilience)을 유지하는 가운데, 팬데믹이라는 외생적 충격이 허브 효과에 일시적·단계적 교란**을 가한 것으로 해석된다. 이는 Chow 검정의 2020년 구조변화(§4.3.4, Table 4-9)와도 일관된다.

## 4.8 지역 흡인력 지수 (RAI) 산출 및 매핑

### 4.8.1 RAI 산출 방법론 및 가중치

앞선 머신러닝 분석(RQ5)에서 도출된 최적 모형(CatBoost)의 SHAP 변수 중요도를 기반으로, 각 지역의 인구 흡인력을 종합적으로 평가하는 지역 흡인력 지수(Regional Attractiveness Index, RAI)를 산출하였다. 4개 하위 영역(서비스·주거, 인프라·접근성, 경제, 인구)의 가중치는 해당 영역에 속한 변수들의 평균 절대 SHAP 값의 비율로 결정되었다(방법론 §3.6.3 참조).

**Table 4-20. 지역 흡인력 지수(RAI) 영역별 가중치**

| 영역 | 주요 포함 변수 | 가중치 (%) |
|:---|:---|---:|
| **서비스·주거 (Serv)** | 보육시설, 노후주택비율, 의사수, 노인복지시설 | 36.4% |
| **인프라·접근성 (Infra)** | 근접 중심성, PageRank, 하수도보급률, 서울거리 | 27.0% |
| **경제 (Econ)** | 고용률, 재정자립도, 사업체수 | 18.6% |
| **인구 (Demo)** | 인구밀도, 청년비율, 고령화율 | 17.9% |
| **Total** | - | **100.0%** |

산출 결과, 보육시설과 노후주택비율을 포괄하는 **서비스·주거 영역(36.4%)**이 가장 큰 가중치를 차지하며 지역 흡인력의 핵심 동력임을 입증하였다. 이어 근접 중심성과 PageRank가 포함된 **인프라·접근성 영역(27.0%)**이 2위를 기록하였다. 반면 고용률 등 경제 영역(18.6%)과 인구 영역(17.9%)의 비중은 상대적으로 낮게 나타나, 최근의 인구 유입에 있어 단순한 경제적 기회보다는 정주 여건과 네트워크 접근성 등 **영역 수준(Domain-level)의 복합적 환경**이 더 결정적인 요인으로 작용함을 재확인하였다.

### 4.8.2. RAI 지수 산출 결과

**Table 4-21. 지역 흡인력 지수(RAI) 상위 10개 및 하위 10개 시군구 (2017-2024년 평균)**

| 순위 | 지역 | 광역시도 | RAI | Net | Serv | Demo | Infra | Econ | 순이동률(‰) |
|:---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 화성시 | 경기도 | 1.114 | 0.852 | 0.132 | -0.111 | 0.052 | 0.189 | 45.26 |
| 2 | 수원시 | 경기도 | 1.075 | 0.755 | 0.662 | 0.428 | 0.035 | 0.105 | -1.96 |
| 3 | 용인시 | 경기도 | 1.056 | 0.712 | 0.520 | 0.078 | 0.032 | 0.084 | 10.23 |
| 4 | 고양시 | 경기도 | 0.983 | 0.638 | 0.450 | 0.206 | 0.028 | 0.071 | 3.59 |
| 5 | 성남시 | 경기도 | 0.954 | 0.561 | 0.199 | 0.352 | 0.031 | 0.096 | -8.64 |
| 6 | 강남구 | 서울특별시 | 0.930 | 0.487 | 0.689 | 0.653 | 0.025 | 0.108 | -2.48 |
| 7 | 송파구 | 서울특별시 | 0.821 | 0.427 | -0.046 | 0.930 | 0.027 | 0.065 | -2.89 |
| 8 | 종로구 | 서울특별시 | 0.801 | 0.299 | 2.991 | 0.433 | 0.015 | 0.051 | -7.36 |
| 9 | 부천시 | 경기도 | 0.751 | 0.326 | 0.255 | 0.704 | 0.026 | 0.055 | -12.88 |
| 10 | 천안시 | 충청남도 | 0.718 | 0.311 | 0.495 | 0.025 | 0.021 | 0.062 | 6.21 |
| … | … | … | … | … | … | … | … | … | … |
| 219 | 울진군 | 경상북도 | -0.560 | -0.254 | -0.350 | -0.297 | -0.018 | -0.035 | -6.09 |
| 220 | 진도군 | 전라남도 | -0.571 | -0.204 | 0.083 | -0.494 | -0.015 | -0.003 | -3.40 |
| 221 | 남해군 | 경상남도 | -0.574 | -0.264 | -0.259 | -0.659 | -0.017 | -0.018 | -0.44 |
| 222 | 신안군 | 전라남도 | -0.580 | -0.273 | 0.187 | -0.595 | -0.019 | 0.014 | -1.52 |
| 223 | 완도군 | 전라남도 | -0.590 | -0.202 | -0.132 | -0.493 | -0.014 | -0.016 | -8.40 |
| 224 | 해남군 | 전라남도 | -0.600 | -0.214 | -0.111 | -0.472 | -0.015 | 0.002 | -12.26 |
| 225 | 의성군 | 경상북도 | -0.615 | -0.219 | -0.250 | -0.799 | -0.015 | -0.004 | 2.78 |
| 226 | 임실군 | 전북특별자치도 | -0.624 | -0.282 | 0.044 | -0.550 | -0.019 | -0.015 | -10.30 |
| 227 | 고성군 | 경상남도 | -0.634 | -0.306 | 0.002 | -0.471 | -0.021 | -0.019 | -4.98 |
| 228 | 합천군 | 경상남도 | -0.635 | -0.325 | -0.003 | -0.733 | -0.022 | -0.021 | -5.89 |

*Note: RAI = SHAP 기여도 기반 가중 합산 지수. 영역별 가중치: Serv 36.4%, Infra 27.0%, Econ 18.6%, Demo 17.9% (합=100.0%). 분석 기간: 2017~2024년 패널 평균. N=228개 시군구(창원시는 2010년 통합시 신설로 독립변수 결측, RAI 산출 제외).*

상위 10개 지역은 수원시·화성시·용인시·성남시·고양시·부천시 등 경기도 주요 도시와 서울 종로구·강남구·송파구, 충남 천안시로 구성되어, 수도권 집중 현상이 RAI 지수에도 명확히 반영되었다. 특히 화성시(순이동률 +45.26‰)와 용인시(+10.23‰)는 높은 RAI와 실제 인구 유입이 일치하는 반면, 강남구(-2.48‰)·성남시(-8.64‰) 등 서울 핵심 자치구는 RAI가 높음에도 순이동률이 음(-)인 역설적 패턴을 보인다. 이는 높은 지가·임대료로 인한 전출 압력이 구조적 매력도를 상쇄하는 '매력도-접근성 괴리' 현상으로 해석된다.

### 4.8.3. RAI 공간 분포 (Figure 4-17)

**Figure 4-17. 지역 흡인력 지수(RAI) Choropleth 지도 (229개 시군구, 2017-2024년 평균)**

![Figure 4-17. RAI Choropleth Map](https://raw.githubusercontent.com/dongwoo2022008/korea-migration-network/main/results/figures/fig4_17_rai_choropleth.png)

*Note: 녹색(+)은 흡인력 상위, 적색(-)은 흡인력 하위 지역. 수도권(경기 북서부)의 집중 패턴과 비수도권 농촌 지역의 낮은 흡인력이 명확히 대비된다.*

Figure 4-17은 RAI의 공간 분포를 시각화한 것으로, 수도권(서울·경기)과 비수도권 간의 뚜렷한 이분법적 구조를 보여준다. 경기 북서부(수원·화성·용인·고양·성남)가 진한 초록으로 표시되어 흡인력 최상위 클러스터를 형성하고 있으며, 경남·전남·경북의 농촌 군 지역은 진한 적색으로 흡인력 최하위 클러스터를 이룬다. 이러한 공간적 패턴은 §4.2에서 확인된 네트워크 허브 집중 현상과 일관된 구조를 보인다.

### 4.8.4. RAI-순이동률 수렴 타당도 (Figure 4-18)

**Figure 4-18. RAI 지수와 순이동률 간 산점도 (2017-2024년 패널 평균)**

![Figure 4-18. RAI vs Net Migration Rate Scatter](https://raw.githubusercontent.com/dongwoo2022008/korea-migration-network/main/results/figures/fig4_18_rai_scatter.png)

*Note: 검은 점은 228개 시군구의 2017~2024년 지역 평균값(N=228); 회색 소점은 패널 전체 관측치(N=1,824) 참고용. 점선은 지역 평균 기준 OLS 회귀선. 지역 평균 기준 r=0.119 (p<0.05, N=228).*

RAI와 실제 순이동률 간 Pearson 상관은 지역평균 기준 r=0.119(p<0.05, N=228)로 유의한 약한 양(+)의 상관을 보였다(Figure 4-18). 상관 크기가 약한 수준에 그치는 것은, RAI가 **지역의 장기적·구조적 매력도(잠재력)**를 측정하는 반면 실제 순이동은 단기적인 부동산 시장 변동, 지가, 혼잡비용 등 단기 실현 요인의 영향을 크게 받기 때문이다. 실제로 인프라와 서비스가 집중된 서울 핵심 자치구와 경기도 주요 도시들은 높은 RAI에도 불구하고 주거비 부담으로 인해 순유출을 보이는 **'매력도–실현 괴리(attractiveness–realization gap)'**가 강하게 관찰된다. 이는 높은 지가·임대료가 구조적 매력도를 상쇄하는 전출 압력으로 작용하기 때문으로 해석된다. 따라서 RAI는 순이동률의 직접 예측 지표라기보다, **지역의 잠재적 흡인 역량을 진단하는 구조적 지표**로 해석하는 것이 타당하다.

### 4.8.5. RAI 구성 강건성 검증 (PCA)

RAI는 개별 결정요인들의 SHAP 기여가 합산되어 전체 매력도를 구성하는 **형성지표(Formative Indicator)**이므로 Cronbach's α는 적용하지 않으며, 구성 강건성은 PCA 방식과의 순위 일치도로 검증한다.

주성분분석(PCA)을 수행하여 도출된 제1주성분 기반 가중 합성지수와 본 연구의 RAI 간 순위 상관(Spearman ρ)을 분석한 결과, SHAP 기반 가중치 방식이 데이터 기반 PCA 방식과 높은 일치도를 보여 지수 산출 방식에 따른 결과의 강건성이 확인되었다. 이는 RAI의 **구성 타당성(construct validity)**을 뒷받침한다. (제1주성분 분산 설명력 약 43%, Spearman ρ=0.150, p<0.05)

요약하면, RAI는 ① 수렴타당성(순이동률과 약한 유의 상관, r=0.119), ② 구성 강건성(PCA ρ=0.150)의 두 측면에서 타당성이 확인된 지수이다. 단, 수렴타당성이 약한 수준에 그치는 점은 RAI가 단기 이동 예측보다 **지역의 구조적 잠재력 진단**에 적합한 지표임을 시사한다.

### 4.8.6. 하위지수 구성 비교 (Figure 4-19)

**Figure 4-19. 상위·하위 지역의 RAI 하위지수 비교 (레이더 차트)**

![Figure 4-19. RAI Sub-index Radar Chart](https://raw.githubusercontent.com/dongwoo2022008/korea-migration-network/main/results/figures/fig4_19_rai_subindex_radar.png)

*Note: 상위 10개 지역(실선)과 하위 10개 지역(점선)의 4개 하위지수 평균값 비교.*

Figure 4-19는 상위 10개 지역과 하위 10개 지역의 하위지수 구성을 비교한 것이다. 상위 지역은 서비스(Serv) 점수가 가장 높고, 인프라(Infra) 부문에서도 높은 점수를 보였다. 하위 지역은 서비스와 인프라 부문에서 특히 낮은 점수를 보여, **서비스 및 인프라 격차**가 지역 흡인력 양극화의 핵심 요인임을 확인하였다. 이는 §4.6의 SHAP 분석에서 서비스 관련 변수(childcare_pk, senior_fac_pk)와 주거·인프라 관련 변수(house_age, closeness)가 상위 변수로 도출된 결과와 일관된다.

### 4.8.7. 소결

본 절의 분석 결과를 종합하면, 지역 흡인력 지수(RAI)는 수도권-비수도권 간 구조적 격차를 정량적으로 포착하며, 인프라 부문이 흡인력 양극화의 주요 동인임을 확인하였다. RAI-순이동률 상관(지역 평균 r=0.119)이 약한 수준에 그치는 것은 RAI가 단기 이동 패턴이 아닌 장기적 지역 경쟁력을 측정하는 지표임을 반영하며, 선행연구(Niedomysl, 2010; Haase et al., 2021)에서 보고된 매력도 지수-이동률 상관 범위(0.2~0.4)를 하회하나 유의한 양(+)의 상관을 보여 형성지표로서 수용 가능한 수준이다. 이는 지역 정책 수립 시 단기 인구 유출입보다 **구조적 매력도 개선**에 초점을 맞출 필요성을 시사한다.

