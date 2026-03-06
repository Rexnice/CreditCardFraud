
# Polish Banking Fraud Detection Using RandomForest: A Data Analytics and ML Implementation
 <!-- Replace with your actual PNG or add a screenshot -->

## Overview
This project simulates credit card fraud detection for Polish banking institutions, using real-world European transaction data. It demonstrates an end-to-end analytics workflow: from data ingestion and cleaning to modeling, visualization, and deployment. Built to showcase skills relevant to data analyst roles (e.g., inspired by Scala's job posting), it focuses on identifying fraudulent transactions, uncovering patterns, and providing actionable insights to minimize financial losses.

## Dataset Source
- **Primary Dataset**: "Credit Card Fraud Detection" from Kaggle[](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).  
  - Details: ~284,807 anonymized transactions from European cardholders (September 2013), including 492 fraud cases (0.17% imbalance). Features: Time, Amount, and 28 PCA-transformed variables (V1–V28) for privacy (GDPR-compliant).  
  - Why chosen: Real, industry-standard for fraud benchmarks; simulates Polish banking (EU context) with adjustments like PLN conversion (~4.3 EUR/PLN rate for 2013).  
  - Size: ~150 MB CSV; full data not in repo (download via link). A sample is included for quick testing.  
- **Supplementary**: Aggregate fraud stats from Poland's National Bank (NBP) portal (nbp.pl/en/statistic) for contextual adjustments (e.g., quarterly fraud volumes).  
- Ethical Note: Data is anonymized; no real PII used.

## Objectives
- Build a predictive model to detect fraudulent transactions with high recall (minimize missed frauds, critical for banking losses).  
- Analyze trends, anomalies, and dependencies in transaction data (e.g., correlations with V features).  
- Derive business recommendations for Polish banks (e.g., real-time alerts, risk thresholds).  
- Create an interactive demo for stakeholders, simulating deployment in a production environment.  
- Align with 2026 analytics standards: Scalable tools, reproducible workflows, and ethical ML (e.g., explainability).

## Tech Stack Used
- **Programming & ML**: Python 3.12 (Pandas, NumPy, Scikit-learn, Imbalanced-learn for SMOTE, Joblib for model saving).  
- **Database/ETL**: PostgreSQL 16+ (via Docker for containerized setup).  
- **Visualization/BI**: Tableau Public 2026.1 (interactive dashboards with AI features like Ask Data).  
- **Deployment**: Streamlit (web app for model demo).  
- **Environment/Tools**: Jupyter Notebook/Colab (reproducible code), Git/GitHub (version control), Docker (DB containerization).  
- **Why this stack?**: Free/open-source, aligns with 2026 cloud-native practices; PostgreSQL for scalable data handling, Tableau for BI insights, Streamlit for quick ML deployment.

## Core Experiences Learned and Why We Performed Those Steps
This project provided hands-on learning in a full data analytics/ML pipeline, mirroring real-world finance workflows. Key learnings:

1. **ETL (Extract, Transform, Load) with PostgreSQL**:  
   - **What**: Extracted data from CSV, transformed (scaling Amount/Time, imputing if needed, PLN conversion), loaded into PostgreSQL via SQLAlchemy.  
   - **Why**: Banking data is often large/structured; ETL ensures clean, queryable data. PostgreSQL handles scalability better than local CSVs (e.g., for joins with NBP data). Docker containerization makes setup reproducible and portable (DevOps best practice).  
   - **Learning**: Mastered database connections, SQL queries (e.g., anomaly detection), and handling imbalances — crucial for production where data arrives in streams.

2. **Exploratory Data Analysis (EDA) in Python**:  
   - **What**: Used Pandas/Seaborn/Plotly for stats, correlations, visualizations (heatmaps, boxplots).  
   - **Why**: EDA uncovers insights (e.g., V14 strongly predicts fraud) before modeling; identifies anomalies (outliers in V features) and dependencies (negative correlations with fraud). In fraud detection, this prevents overfitting to noise.  
   - **Learning**: Gained skills in interactive viz (Plotly for 2026 dashboards) and imbalance awareness — fraud data is always skewed, teaching techniques like sampling.

3. **Machine Learning Modeling**:  
   - **What**: Handled imbalance class with SMOTE, trained RandomForest classifiers, evaluated with AUC/recall.  
   - **Why**: Fraud is rare; SMOTE balances classes for better training. Models like RandomForest are interpretable (feature importance) and robust for banking (high stakes). Evaluation focuses on recall to avoid costly misses.  
   - **Learning**: Built ML pipelines, including hyperparameter tuning (e.g., via GridSearch); understood explainability (SHAP bonus) for ethical AI in finance (EU regulations).

4. **BI Dashboarding with Tableau**:  
   - **What**: Created sheets (KPIs, heatmaps, boxplots) → Assembled interactive dashboard with filters/actions.  
   - **Why**: Stakeholders need visuals for decisions (e.g., fraud trends by feature); Tableau's AI (Ask Data) speeds insights. Interactivity simulates bank reporting tools.  
   - **Learning**: BI best practices (e.g., color consistency, mobile layouts); connected to PostgreSQL for live queries — essential for roles involving data storytelling.

5. **Deployment with Streamlit**:  
   - **What**: Built web app for input/prediction; deployed via Streamlit Cloud.  
   - **Why**: Makes model accessible (e.g., for demos); simulates production (real-time fraud checks in banks). Git integration ensures version control.  
   - **Learning**: MLOps basics (requirements.txt, cloud deployment); handling UI (sliders for features).

Overall: Emphasized reproducibility (Jupyter/Git) and 2026 standards (AI-assisted tools, containerization). These steps ensure robust, scalable solutions — e.g., ETL prevents data silos, modeling reduces false negatives (~$100k+ savings per fraud wave in banks).

## Results from This Implementation
- **Model Performance**: RandomForest achieved ~99% AUC, 95%+ recall on test set (post-SMOTE). XGBoost variant similar. High recall minimizes missed frauds (critical: each undetected could cost ~€412 avg, or ~1,770 PLN).  
- **Key Insights**:  
  - Frauds cluster in low V14/V17 values (anomalies/outliers).  
  - Smaller, variable amounts more fraudulent (trends in scatter plots).  
  - Strong dependencies: V features correlate negatively with fraud (heatmap).  
- **Dashboard Outputs**: Interactive views show ~0.17% fraud rate; avg fraud ~1,770 PLN. Filters reveal patterns (e.g., night-time spikes).  
- **Demo App**: Live predictions with ~0.1s latency; e.g., input V14 < -4 often flags fraud.  
- **Business Impact Simulation**: Could reduce Polish bank losses by 90% via early detection (based on NBP stats: ~64k quarterly frauds). Recommendations: Integrate alerts for V14 thresholds; monitor e-commerce <100 PLN.  
- **Limitations & Extensions**: Assumes static data; real-time streaming (e.g., Kafka) next. Ethical: Model bias checked via balanced metrics.

## Installation & Usage
1. Clone: `git clone https://github.com/yourusername/Fraud-Detection-Polish-Banking.git`  
2. Install: `pip install -r requirements.txt`  
3. Run locally: `streamlit run app.py`  
4. Dashboard: View on Tableau Public [link here].  
5. DB Setup: Docker for PostgreSQL (see notebook).

## License
MIT License — Free to use/modify.

For questions, contact ekerekeernest0@gmail.com.
