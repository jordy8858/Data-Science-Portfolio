# Predictive Lead Scoring & ROI Optimization

## Business Problem
In high-volume automotive digital retail, Business Development Centers (BDCs) and internet sales teams face massive inefficiencies from low-intent digital leads. Pushing every inbound CRM lead equally down the sales funnel results in thousands of wasted labor hours chasing non-converting prospects, drastically increasing customer acquisition costs (CAC).

## Solution
This project implements a self-contained machine learning pipeline using an **XGBoost Classifier** to analyze modern digital retail CRM metrics. The model evaluates a prospective buyer's real-time digital behavior—such as time spent on vehicle configurators, trade-in valuations, and financing application starts—to generate a data-driven probability score. Instead of treating all leads equally, the sales floor can prioritize outreach based on predictive conversion likelihoods.

## Key Features
* **Synthetic CRM Data Engine:** Simulates realistic website behaviors and categorical consumer metrics (credit tiers, lead sources) reflecting modern dealership lead flow.
* **Imbalance Handling:** Utilizes advanced XGBoost hyperparameter tuning (`scale_pos_weight`) to natively handle heavily skewed conversion rates (88% non-converters vs. 12% converters).
* **Corporate ROI Calculator:** Translates standard technical metrics (ROC-AUC, F1-Score) into a tangible dollar-value payroll optimization matrix.

## Financial & Operational Impact
* **Conversion Capture:** Prioritizing just the top 20% of highest-scoring leads successfully captures **87.4% of total actual conversions**.
* **BDC Efficiency:** The model automatically eliminates the need for salespeople to chase **1,559 dead leads** per batch.
* **Labor Cost Savings:** Reduces cold outreach overhead, resulting in an estimated **$19,487.50 in saved BDC labor costs** per 2,000 leads analyzed.

## Tech Stack
* **Language:** Python
* **Modeling:** XGBoost, Scikit-Learn
* **Data Engineering:** Pandas, NumPy
* **Explainability:** SHAP (SHapley Additive exPlanations), Seaborn, Matplotlib

## How to Run
1. Ensure you have the dependencies installed: `pip install xgboost sklearn pandas numpy shap matplotlib seaborn`
2. Open the notebook `Lead_Scoring_Pipeline.ipynb` in your IDE (e.g., PyCharm) or Google Colab.
3. Run all cells to execute the simulation, model training, SHAP feature evaluation, and ROI calculation.
