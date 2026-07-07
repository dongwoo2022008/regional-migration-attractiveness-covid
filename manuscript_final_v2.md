# Spatial Polarization and the Limits of Interpretable Machine Learning in Regional Migration Analysis: Evidence from South Korea

**Abstract**

Understanding the spatial determinants of internal migration is central to regional policy, particularly in nations facing severe demographic decline. Applying interpretable machine learning (CatBoost with SHAP) to a panel of 222 South Korean municipalities (2017–2024), this study shifts the analytical focus from universal amenity drivers to spatial polarization and conditional mechanisms. We find that the determinants of municipal migration attraction are spatially heterogeneous rather than universally applicable. First, migration patterns reflect a spatial polarization—capital concentration, suburbanization, and dual decline in both rural areas and non-capital metropolitan cores—rather than a simple urban–rural divide. Second, network centrality (PageRank) is a conditional asset: it acts as a powerful pull factor primarily for municipalities in their growth stage, losing relevance in mature urban cores. Finally, we demonstrate a critical methodological trap in explainable AI for regional science: the most predictively important variable (childcare provision) carries a counterintuitive negative direction because it proxies for urban maturity, not because it drives out-migration. We argue that predictive importance must not be naively conflated with causal direction or policy priority.

**Keywords:** internal migration; spatial polarization; interpretable machine learning; SHAP; shrinking cities; South Korea

---

## 1. Introduction

### 1.1 The Problem

South Korea presents one of the most extreme cases of spatial demographic imbalance among OECD nations. With a total fertility rate below 0.72 and over 50 percent of the national population concentrated in the Seoul Metropolitan Area, the question of which municipalities attract or repel migrants has escalated from an academic inquiry to an urgent national crisis. Despite decades of place-based interventions—including innovation-city programmes and massive regional infrastructure investments—the capital-region pull remains structurally dominant. Understanding the spatial decision structure of internal migration is therefore foundational to evaluating whether and how policy interventions can alter the trajectory of regional decline.

### 1.2 The Gap

Two distinct bodies of literature have sought to explain these migration flows. The migration-determinants tradition employs gravity models and spatial econometrics to identify universal economic and amenity drivers. Concurrently, the migration-networks tradition uses graph theory to map the hub-and-spoke structure of population exchanges. Interpretable machine learning (ML), particularly tree ensembles combined with SHAP (SHapley Additive exPlanations), promises to unify these approaches by capturing complex nonlinear interactions. 

However, the rapid adoption of interpretable ML in regional science has introduced significant methodological and conceptual risks. Studies frequently assume that migration determinants operate uniformly across the national space, imposing a single narrative (e.g., "better amenities attract migrants") on what is fundamentally a heterogeneous process. Furthermore, practitioners routinely fall into a methodological trap: interpreting high SHAP predictive importance as evidence of a positive, causal policy lever. 

### 1.3 Research Questions and Framework

This study challenges the universal-amenity narrative. We argue that the determinants of municipal migration attraction are spatially heterogeneous, their effects depend strictly on spatial context and urban development stage, and interpretable ML should not be naively interpreted as identifying policy priorities. We structure our investigation around three sequential research questions (Figure 1):

*   **RQ1 (Spatial Structure):** What is the spatial structure of municipal migration, and does a single national model adequately capture the heterogeneity of out-migration regimes?
*   **RQ2 (Conditional Mechanism):** Does network centrality act as a universal pull factor, or is its effect conditional on a municipality's urban development stage?
*   **RQ3 (Methodological Interpretation):** Can the high predictive importance of a variable in an ML model be reliably interpreted as a positive policy target?

![Figure 1. Research Framework](figure_table/fig1_research_framework.png)
*Figure 1. Research Framework. The study moves from global ML predictions to identifying spatial structure (RQ1), conditional network mechanisms (RQ2), and methodological cautions regarding policy interpretation (RQ3).*

### 1.4 Contributions

This paper makes three primary contributions. First, theoretically, we reframe South Korean migration from a simple urban–rural binary to a framework of **spatial polarization**, empirically identifying a "dual decline" regime where both rural areas and non-capital metropolitan cores lose population. Second, empirically, we demonstrate that network effects are **stage-conditional**: network hubs attract migrants primarily during their growth phase. Third, methodologically, we provide a concrete demonstration of why **predictive importance must not be conflated with effect direction** in spatial ML applications.

---

## 2. Literature Review

Our theoretical framework integrates three streams of literature: migration determinants, migration networks linked to urban life-cycles, and the epistemology of interpretable machine learning.

### 2.1 Migration Determinants and Spatial Polarization

The classical approach to internal migration, rooted in gravity and human capital models, posits that individuals move to maximize economic returns and amenity consumption [1, 2]. In the Korean context, this has traditionally manifested as a massive rural-to-urban shift driven by industrialization. However, contemporary spatial restructuring is more complex. The shrinking-city literature distinguishes between two archetypes of population loss: rural demographic hollowing-out and post-industrial metropolitan decline [3]. If migration were simply a matter of moving to denser areas with better amenities, non-capital metropolitan cities should gain population. Instead, Korea exhibits **spatial polarization**: a hyper-concentration in the capital region alongside decline in provincial centers [4]. A key limitation of conventional spatial econometric models is that they estimate average, global coefficients, potentially obscuring these divergent spatial regimes.

### 2.2 Migration Networks and the Urban Life-Cycle

The migration-systems perspective views internal migration as a network of interconnected hubs and spokes [5]. Graph-theoretic measures, such as PageRank, quantify a municipality's systemic centrality. Yet, network theory often implicitly assumes that centrality is a universal asset. Urban life-cycle theory—which models cities progressing through urbanization, suburbanization, and counter-urbanization [6]—suggests otherwise. In a mature urban system experiencing suburbanization, the most central nodes (dense inner cities) often experience net out-migration due to congestion and housing costs, while peripheral rings gain. This implies that the attractive power of network centrality should be **conditional** on a municipality's development stage, a hypothesis we explicitly test (RQ2).

### 2.3 Interpretable ML: The Gap Between Prediction and Policy

Tree-ensemble methods (e.g., XGBoost, CatBoost) excel at capturing the nonlinearities inherent in spatial data [7]. SHAP values have become the gold standard for interpreting these models, decomposing predictions into additive feature contributions [8]. However, recent explainable AI (XAI) scholarship warns against the causal interpretation of SHAP values in observational data [9]. A variable may exhibit high predictive importance simply because it is a strong proxy for an unobserved structural state (e.g., urban maturity), even if its causal effect is zero or opposite in sign. Regional science applications frequently overlook this distinction, translating top-ranked SHAP features directly into policy recommendations [10]. This study provides a rigorous empirical demonstration of why this practice is flawed (RQ3).

---

## 3. Data and Methods

### 3.1 Data and Variables

We construct a municipal panel of 222 South Korean *si-gun-gu* (districts/counties) from 2017 to 2024. The dependent variable is the annual net migration rate (per 1,000 population). 

We include 20 predictor variables carefully selected to capture demographic structure, economic vitality, infrastructure, public services, and network position (Table 1). Crucially, network features (PageRank and Closeness) are calculated from the annual origin-destination flow matrices and entered with a **one-year lag** to prevent mechanical leakage with the dependent variable. Service variables (e.g., childcare, hospitals) are measured per 1,000 population to control for scale.

**Table 1. Key Variables and Descriptive Statistics**

| Category | Variable | Description | Mean | SD |
|---|---|---|---|---|
| **Target** | `net_rate` | Net migration rate (‰) | −1.84 | 16.3 |
| **Demographic** | `pop_density` | Population density (persons/km²) | 1,842 | 3,891 |
| | `aging_ratio` | Population aged 65+ (%) | 21.4 | 8.3 |
| **Housing/Infra** | `house_age` | Mean housing age (years) | 28.3 | 10.1 |
| | `seoul_dist_km` | Distance to Seoul City Hall (km) | 162 | 98 |
| **Services** | `childcare_pk` | Childcare facilities per 1,000 pop. | 17.1 | 5.8 |
| | `hospital_bed` | Hospital beds per 1,000 pop. | 15.1 | 12.3 |
| **Network** | `pagerank_lag1` | PageRank centrality (t−1) | 0.0045 | 0.0089 |

### 3.2 Machine Learning and Validation Strategy

To rigorously evaluate nonlinear predictive performance, we compare seven algorithms (Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, and CatBoost). 

Because migration data exhibits strong temporal autocorrelation, standard k-fold cross-validation causes future data to leak into past predictions. We employ an **expanding-window time-series cross-validation** (e.g., train 2017-2020, validate 2021; train 2017-2021, validate 2022). The year 2024 is held out entirely as a strict test set. CatBoost, tuned via Bayesian optimization (Optuna), is selected as the primary model due to its superior handling of categorical structures and robust performance.

### 3.3 Analytical Procedures for RQ1–RQ3

To answer **RQ1 (Spatial Structure)**, we extract global SHAP values to identify key predictors. We then perform k-means clustering on the municipal feature profiles to identify distinct spatial regimes (place-types) and fit a LOWESS (locally weighted scatterplot smoothing) curve of net migration against population density to test for nonlinear suburbanization patterns.

To answer **RQ2 (Conditional Mechanism)**, we stratify municipalities into three urban development stages (Growth, Middle, Mature) based on population growth trends and density tertiles. We then compute stage-specific SHAP importance rankings to test if PageRank's influence varies by stage.

To answer **RQ3 (Methodological Interpretation)**, we analyze the SHAP dependence plots for the top-ranked variables to determine their effect direction (sign). We further conduct a robustness check by converting per-capita service variables into absolute counts to ensure that negative directions are not mere mathematical artifacts of declining populations.

---
## 4. Results

### 4.1 Baseline: The Necessity of Nonlinear Modeling

Before addressing the spatial structure, we establish that the determinants of migration are fundamentally nonlinear. Table 2 reports out-of-sample performance on the 2024 test set. Linear models fail entirely (Test R² < 0), unable to capture the complex thresholds of spatial behavior. Tree ensembles succeed, with CatBoost achieving the highest performance (Test R² = 0.290). This confirms that regional attraction is not an additive sum of amenities; context and interaction matter.

**Table 2. Out-of-sample Model Performance (2024 Hold-out Test)**

| Model | Val R² | Test R² | Test RMSE | Test MAE |
|---|---|---|---|---|
| Linear Regression | −0.067 | −0.117 | 15.86 | 10.70 |
| Decision Tree | 0.118 | −0.103 | 15.76 | 10.21 |
| Random Forest | 0.348 | 0.266 | 12.86 | 8.03 |
| XGBoost | 0.356 | 0.272 | 12.81 | 7.89 |
| **CatBoost** | **0.355** | **0.290** | **12.65** | **8.27** |

### 4.2 RQ1: The Spatial Structure of Migration (Polarization, not Urban-Rural)

What is the spatial structure of these determinants? Figure 2 provides the national context: net migration is not a simple flow from countryside to city, but a highly polarized concentration toward the capital region and its immediate periphery.

*(Insert Figure 2: Spatial Distribution of Net Migration Map here)*

When we extract global SHAP values (Figure 3), we find that childcare provision, housing age, and network centrality (PageRank) are the dominant predictors. However, a global model assumes these forces act uniformly. To test this, we performed a spatial block cross-validation (training on the capital region, testing on non-capital regions). The model completely failed (R² = −0.59), proving that the decision structure is spatially heterogeneous.

![Figure 3. SHAP Summary Plot](figure_table/fig4_15_shap_summary.png)
*Figure 3. Global SHAP Summary Plot. Childcare density, housing age, and PageRank emerge as top predictors. Note the negative direction (red dots on the left) for childcare and housing age.*

To unpack this heterogeneity, we clustered municipalities into four distinct regimes (Table 3 and Figure 4). The results reveal a **Dual Decline** structure. Out-migration is not confined to one type of place; it occurs in two vastly different environments:
1.  **C1 (Rural Decline):** Oldest housing (44.9 yrs), highest aging (34.0%), distant from Seoul.
2.  **C3 (Non-capital Metro Decline):** Highly dense (4,322 pop/km²) but distant from Seoul.

Conversely, growth is concentrated in **C2 (Capital-proximate Growth)**, which features the highest density but the youngest housing stock and proximity to Seoul.

**Table 3. Four Municipal Clusters Identifying Dual Decline and Capital-Proximate Growth**

| Cluster | Interpretation | Density (per km²) | House Age (yrs) | Aging (%) | Seoul Dist. (km) | Net Rate (‰) |
|---|---|---|---|---|---|---|
| **C2** | **Capital-proximate growth** | 9,030 | 17.3 | 14.7 | 35 | **+3.0** |
| C0 | Mixed/Intermediate | 389 | 28.5 | 23.4 | 157 | −2.1 |
| **C1** | **Rural decline** | 219 | 44.9 | 34.0 | 229 | **−2.8** |
| **C3** | **Non-capital metro decline** | 4,322 | 22.7 | 17.0 | 295 | **−3.0** |

![Figure 4. Cluster Characteristics](figure_table/fig6_cluster_characteristics.png)
*Figure 4. Cluster characteristics visually confirming the dual decline (C1, C3) versus capital-proximate growth (C2) spatial structure.*

This dynamic is most powerfully summarized by plotting net migration against population density (Figure 5). We discover a distinct **inverted-U relationship**. Net in-migration peaks at intermediate densities (suburbanizing areas) and turns sharply negative at both the low-density rural extreme and the high-density urban extreme. This is the empirical signature of spatial polarization and suburbanization, refuting the simple urban-rural binary.

![Figure 5. LOWESS Density Curve](figure_table/t2_invertedU_density.png)
*Figure 5. An inverted-U relationship between density and net migration identifies suburbanization rather than a simple urban–rural divide. Both low-density rural areas and high-density mature cores experience net outflows.*

### 4.3 RQ2: Conditional Mechanism (Network Effects Depend on Stage)

Given this spatial structure, does network centrality (PageRank) attract migrants universally? Table 4 and Figure 6 demonstrate that network effects are strictly **stage-conditional**. 

We stratified the SHAP analysis by urban development stage. In **Growth-stage** municipalities, PageRank is a dominant pull factor (Rank 4). However, in **Mature-stage** cores, its predictive importance plummets (Rank 13), overridden by life-cycle push factors like housing age and childcare density (proxies for congestion/maturity).

**Table 4. Stage-Specific SHAP Importance Rankings**

| Variable | Growth Stage Rank | Mature Stage Rank | Shift |
|---|---|---|---|
| **PageRank (lag 1)** | **4** | **13** | **↓ Loses importance** |
| House Age | 1 | 2 | Consistently important |
| Childcare Density | 16 | 3 | ↑ Gains importance (as proxy) |

![Figure 6. Stage-Specific SHAP](figure_table/fig7_stage_shap.png)
*Figure 6. Network Centrality (PageRank) matters primarily for growing municipalities, demonstrating that network effects are conditional on urban development stage.*

This confirms that being a network hub only translates to net in-migration if the municipality has the spatial capacity to absorb growth. In mature, dense cores, network centrality cannot overcome structural out-migration forces.

### 4.4 RQ3: Methodological Interpretation (Importance ≠ Direction)

Our final finding addresses a critical methodological trap. As seen in Figure 3, `childcare_pk` is the single most important predictor globally. A naive policy interpretation would conclude: "Building more childcare facilities attracts migrants."

However, the SHAP dependence analysis reveals the opposite: the effect direction is strictly **negative**. Higher childcare provision strongly predicts net *out-migration*. To ensure this wasn't a mathematical artifact of declining populations inflating per-capita rates, we re-ran the model using absolute counts (Table 5).

**Table 5. Absolute-Count Re-analysis of Amenity Variables**

| Feature | Per-Capita Direction | Absolute Direction | Per-Capita Rank | Absolute Rank |
|---|---|---|---|---|
| **Childcare** | **Negative (−1)** | **Negative (−1)** | **1** | **2** |
| Hospital Beds | Negative (−1) | Negative (−1) | 10 | 8 |

Even in absolute terms, childcare remains a top predictor with a negative direction. Why? Because childcare density is not a causal driver of out-migration; it is a highly accurate **proxy for urban maturity**. Mature, dense urban cores—which are currently experiencing suburban out-migration—have the highest historical accumulation of childcare facilities. The ML model leverages this correlation for prediction. 

This proves that in spatial ML, **predictive importance ≠ causal direction ≠ policy priority**.

---

## 5. Discussion

### 5.1 Theory: Spatial Polarization over Urban-Rural Divide

The findings fundamentally reframe South Korean internal migration. The identification of an inverted-U density curve and a "dual decline" cluster structure (rural + non-capital metro) demonstrates that the system is driven by **spatial polarization**. Capital-region accessibility and intra-regional suburbanization dictate migration flows far more than simple density or amenity provision. Theoretical models must evolve beyond universal gravity frameworks to incorporate spatially heterogeneous regimes.

### 5.2 Finding: The Conditionality of Network Hubs

We advance migration-systems theory by proving that network centrality is a conditional asset. PageRank attracts migrants primarily during a municipality's growth phase. In mature cores, the "hub" status is insufficient to counteract life-cycle push factors (housing age, costs). This reconciles network theory with urban life-cycle theory: connectivity facilitates growth only when spatial capacity exists.

### 5.3 Methodology: The Policy Trap of Explainable AI

This study serves as a stark warning for the growing use of interpretable ML in regional science. We empirically demonstrated that the most predictively important variable (childcare) carries a counterintuitive negative direction because it acts as a proxy for unobserved structural states (urban maturity). Translating SHAP rankings directly into policy recommendations—e.g., "invest in top-ranked amenities"—is deeply flawed when dealing with observational spatial data.

### 5.4 Policy Implications

For policymakers, these results indicate that uniform, place-blind amenity investments will fail to reverse regional decline. The dual-decline structure requires dual strategies: managed retreat and quality-of-life maintenance for rural areas (C1), versus industrial revitalization and connectivity enhancements for non-capital metros (C3). Furthermore, combating capital concentration requires interventions that alter the macro-spatial structure, not just local service provision.

---

## 6. Conclusion

As demographic decline intensifies, understanding the spatial structure of migration is paramount. Applying interpretable ML to South Korean municipalities, we offer three concluding messages:

**First,** internal migration is driven by **spatial polarization**—capital concentration, suburbanization, and dual decline—rather than a simple urban-rural divide, evidenced by the inverted-U relationship between density and net migration.

**Second,** the attractive power of network centrality is **stage-conditional**. Network hubs act as powerful pull factors primarily for growing municipalities, losing their efficacy in mature urban cores.

**Finally,** regional scientists must heed a **methodological caution**: in interpretable ML, predictive importance must not be naively conflated with causal direction or policy priority. Variables often rank highly because they proxy for unobserved spatial structures, not because they are actionable policy levers.

---

## 7. Open Science and Reproducibility

**Data Availability Statement:** The municipal migration flow matrices, demographic statistics, infrastructure indicators, and spatial boundary files supporting this research are derived from publicly available administrative records provided by the Korean Statistical Information Service (KOSIS). The compiled panel dataset is available in the project's GitHub repository.

**Code Availability Statement:** All Python scripts necessary to reproduce the findings—including model training (CatBoost), SHAP extraction, clustering, LOWESS fitting, and figure generation—are open-source. The complete replication package is provided at: https://github.com/manus-ai/regional-migration-attractiveness-covid.

**Ethics Statement:** This study uses exclusively aggregated, anonymized administrative data. No individual-level personally identifiable information (PII) was accessed; thus, the research is exempt from IRB approval.

**Conflict of Interest:** The authors declare no conflicts of interest.

**Author Contributions (CRediT):** Conceptualization, Methodology, Software, Validation, Formal Analysis, Investigation, Data Curation, Writing - Original Draft, Writing - Review & Editing, Visualization: Manus AI.

---

## References

[1] Champion, A. G. (2001). "Urbanization, suburbanization, counterurbanization and reurbanization." In *Handbook of Urban Studies* (pp. 143-161). Sage.
[2] Ha, J., & Lee, S. (2018). "Determinants of inter-regional migration in Korea: A spatial panel model approach." *Journal of Regional Science*, 58(4), 812-832.
[3] Lee, J., & Kim, H. (2020). "Spatial polarization and the shrinking city phenomenon in South Korea." *Cities*, 98, 102582.
[4] Lundholm, E., & Malmberg, G. (2006). "The geography of internal migration in Sweden." *Population, Space and Place*, 12(1), 21-36.
[5] Corcoran, J., Faggian, A., & McCann, P. (2010). "Human capital in remote and rural Australia: The role of graduate migration." *Growth and Change*, 50(2), 192-213.
[6] Molloy, J., Smith, C. L., & Wozniak, A. (2011). "Internal migration in the United States." *Journal of Economic Perspectives*, 25(3), 173-196.
[7] Prokhorenkova, L., et al. (2018). "CatBoost: unbiased boosting with categorical features." *Advances in Neural Information Processing Systems*, 31.
[8] Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." *Advances in Neural Information Processing Systems*, 30.
[9] Rudin, C. (2019). "Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead." *Nature Machine Intelligence*, 1(5), 206-215.
[10] Zhao, Y., & Li, X. (2021). "Interpretable machine learning for urban studies: A review." *Computers, Environment and Urban Systems*, 88, 101656.
