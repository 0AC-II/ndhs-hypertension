# Sociodemographic Predictors of Hypertension Risk Among Nigerian Adults
## A Machine Learning Analysis of Nigeria DHS 2024

---

## Project Overview

This project builds a machine learning model to estimate hypertension risk among
Nigerian adults using purely sociodemographic features -- no clinical measurements
required. It uses the Nigeria Demographic and Health Survey (NDHS) 2024, the most
recent nationally representative health survey available for Nigeria.

The core question: **Can we identify which unscreened Nigerians are most likely
to be hypertensive, using only information we already know about them?**

---

## Why This Matters

Hypertension is the leading modifiable risk factor for cardiovascular disease in
sub-Saharan Africa. In Nigeria, a large proportion of hypertensive adults have
never been screened. Without a diagnosis, there is no treatment. Without treatment,
the condition progresses silently.

This model is not a clinical diagnostic tool. It is a **population-level screening
signal** -- designed to help health programs prioritize outreach to communities
and demographic groups where undetected hypertension is most likely concentrated.

---

## Data Source

- **Survey:** Nigeria Demographic and Health Surveys (NDHS) 2024
- **Phase:** DHS-VIII (Standard DHS)
- **Files used:** Individual Women's Recode (IR) + Men's Recode (MR)
- **Access:** DHS Program (registration required) -- dhsprogram.com
- **Combined sample:** 51,254 adults (women 15-49, men 15-59)
- **Screened population (model training):** 24,333 adults
- **Unscreened population (risk inference):** 26,673 adults

---

## Target Variable

- **chd02:** Ever been told you have high blood pressure or hypertension
- Applied only to adults who confirmed they had been screened (chd01 = 1)
- Positive rate in screened population: **13.7%**

---

## Features Used

| Variable | Description |
|----------|-------------|
| Age | Respondent's current age |
| Geopolitical Zone | Nigeria's 6 geopolitical zones |
| Total Children Ever Born | Parity (number of births) |
| Wealth Index | Household wealth quintile (1-5) |
| Years of Education | Continuous years of schooling |
| Reads Newspaper | Frequency of newspaper reading |
| Sex | Male / Female |
| Urban/Rural Residence | Type of place of residence |
| Education Level | Categorical education attainment |
| Worked in Last 12 Months | Employment status |
| Sex of Household Head | Male / Female headed household |

---

## Model Performance

| Model | CV AUC (5-fold) | Test AUC |
|-------|----------------|----------|
| Gradient Boosting | 0.6630 ± 0.0105 | **0.6694** |
| Logistic Regression | 0.6574 ± 0.0076 | 0.6631 |
| Random Forest | 0.6069 ± 0.0125 | 0.6015 |

**Final model: Gradient Boosting (sklearn GradientBoostingClassifier)**

AUC of 0.67 using only sociodemographic features -- no blood pressure readings,
no BMI, no laboratory values -- is consistent with comparable population-level
hypertension screening models in the literature.

---

## Key Findings

### 1. Age is the strongest individual predictor
SHAP analysis confirms age as the dominant driver, with SHAP values extending
to +2.5 for the oldest respondents. Clinically expected, but the magnitude
relative to all other features is notable.

### 2. Geopolitical Zone is the second strongest predictor
Geography within Nigeria carries more predictive weight than wealth, education,
or urban/rural status. This is the project's most policy-relevant finding.

### 3. North-South gradient in diagnosed hypertension

| Zone | Screened Adults | Diagnosed Rate |
|------|----------------|----------------|
| North East | 3,611 | **16.7%** |
| North Central | 4,659 | 14.8% |
| North West | 4,317 | 14.4% |
| South South | 4,160 | 14.2% |
| South West | 4,087 | 11.1% |
| South East | 3,499 | **10.4%** |

National average: 13.7%

### 4. Estimated risk among unscreened adults mirrors the pattern

| Zone | Unscreened Adults | Mean Predicted Risk | High-Risk (>15%) |
|------|------------------|--------------------|--------------------|
| North East | 4,570 | 12.8% | **32.3%** |
| North West | 7,887 | 10.1% | 17.4% |
| South South | 3,114 | 9.7% | 15.8% |
| North Central | 5,436 | 9.5% | 15.7% |
| South West | 2,602 | 8.1% | 11.3% |
| South East | 3,064 | 7.3% | 8.0% |

**1 in 3 unscreened adults in the North East is estimated to be at high
hypertension risk and has never had their blood pressure checked.**

### 5. Higher parity predicts higher risk
Total children ever born ranks third in feature importance, carrying signal
independent of age -- likely reflecting cumulative cardiovascular stress
across pregnancies.

---

## Limitations

- Target variable is self-reported diagnosis, not measured blood pressure.
  People who have never been screened cannot report a diagnosis, which is
  precisely why the unscreened inference problem matters.
- BMI was excluded due to 64% missingness (biomarker subsample only).
- Men are underrepresented (12,204 vs 39,050 women) reflecting DHS sampling design.
- Model is associative, not causal. Geopolitical zone captures unmeasured
  structural variables (diet, health system density, conflict exposure) rather
  than geography itself.
- Survey weights were not applied in this version. A weighted analysis would
  improve population-level representativeness.

---

## Repository Structure
```
ndhs-hypertension/
├── data/
│   └── raw/          # DHS files (not included -- requires DHS registration)
├── notebooks/
│   └── analysis.ipynb
├── outputs/
│   ├── roc_curves.png
│   ├── shap_summary.png
│   ├── shap_importance.png
│   ├── zone_prevalence.png
│   └── unscreened_risk_scores.csv
└── README.md
```

---

## How to Reproduce

1. Register at dhsprogram.com and request Nigeria 2024 Standard DHS datasets
2. Download NGIR8BDT.ZIP (Individual Recode) and NGMR8BDT.ZIP (Men's Recode)
3. Unzip into data/raw/
4. Run notebooks/analysis.ipynb top to bottom

Dependencies: pandas, numpy, pyreadstat, scikit-learn, shap, matplotlib
```bash
pip install pandas numpy pyreadstat scikit-learn shap matplotlib
```

---

## Author

**Anthony Okeibuno**
Health Data Systems Strategist | Independent Researcher
Ibadan, Nigeria
