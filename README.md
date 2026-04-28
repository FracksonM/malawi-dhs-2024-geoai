# Malawi Child Stunting Prediction: Survey-Weighted Machine Learning (2024 MDHS)

**Author:** Frackson Makwangwala  
**Contact:** fracksonmakwangwala@gmail.com

## Overview
Survey-weighted multi-algorithm machine learning for nationally representative 
prediction and district-level decomposition of child stunting risk in Malawi, 
using the 2024 Malawi Demographic and Health Survey (n = 5,122 children, 
767 enumeration area clusters).

## Key results
- Best model: Random Forest, ROC-AUC = 0.6856 (95% CI: 0.666, 0.705)
- Survey-weighted stunting prevalence: 37.5% (95% CI: 35.9%, 39.1%)
- Top predictor: maternal height (mean absolute SHAP = 0.0558)
- District SHAP decomposition: 3 risk archetypes across 32 districts

## Notebooks
| Notebook | Description |
|---|---|
| 10 | Survey-weighted model training: RF, XGBoost, LightGBM, LR |
| 10b | Model enhancement: calibration, DeLong test, subgroup analysis |
| 11 | SHAP global analysis and district decomposition |
| 12 | Wasting and anemia models |
| 13 | Spatial GeoAI: GPS integration, spatial lag, environmental covariates |

## Application
`app.py` — Streamlit prediction tool with HSA, DHO, and Researcher dashboards.

```bash
streamlit run app.py
```

## Citation
Makwangwala F. (2025). Survey-Weighted Machine Learning for Nationally 
Representative Prediction and District-Level Decomposition of Child Stunting 
Risk in Malawi: A Multi-Algorithm Analysis of the 2024 Demographic and Health 
Survey. 

## Data access
Raw DHS microdata are not included in this repository (licence restricted).  
Access the 2024 MDHS at https://dhsprogram.com/data/
