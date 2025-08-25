# Intrusion Detection System (IDS)

This project implements a machine learning–based Intrusion Detection System (IDS) designed to detect anomalies in network traffic, specifically distinguishing between BENIGN and DDoS flows. The system provides a complete pipeline covering data preprocessing, feature engineering, model training, experiment tracking, deployment with FastAPI and Docker, and a user-facing Streamlit interface.

# Features

The project includes preprocessing and analysis of the CICIDS2017 dataset, handling missing, infinite, or imbalanced values, and evaluating multiple machine learning models including KNN, Logistic Regression, Random Forest, and Gradient Boosting. Experiment tracking is supported using MLflow. The trained model is deployed through a FastAPI backend, while a Streamlit frontend allows both single and batch predictions. Docker and Docker Compose are used for deployment, and GitHub Actions provides CI/CD automation.

# Dataset

The IDS was trained and tested on the CICIDS2017 dataset. This dataset contains multiple types of cyberattacks, but in this project the focus was on binary classification between BENIGN traffic and DDoS traffic. This subset offers a realistic and high-impact use case for demonstrating an applied IDS.

# Project Workflow

> Data Preparation

The dataset was cleaned by encoding categorical variables, removing irrelevant or ID-like columns, scaling numerical features, and replacing missing or infinite values.

> Exploratory Data Analysis

The structure and class balance of the dataset were examined. Feature distributions were analyzed, and correlation heatmaps were used to identify multicollinearity. Highly correlated features were reduced to improve model generalization.

> Modeling

Multiple models were implemented, including KNN, Logistic Regression, Random Forest, and Gradient Boosting. Metrics such as accuracy, precision, recall, F1-score, and ROC-AUC were compared. Random Forest was selected for deployment as it provided the best balance between accuracy and interpretability.

> Model Deployment

The trained model and feature schema were saved, and a FastAPI backend was built with endpoints for health checks, single predictions, and batch predictions. Input data is validated against the stored feature schema.

> User Interface

A Streamlit application was developed to allow manual input for single predictions and batch CSV uploads. Predictions can be viewed directly in the interface and downloaded for further use.

> Automation and Reproducibility

The backend and frontend were containerized with Docker. Services are defined in docker-compose.yml to ensure consistent deployment across environments. GitHub Actions CI/CD pipeline was added to automate builds, run tests, and verify functionality.

# How to Run

> Clone the repository

    git clone https://github.com/yashpalande26/network-security-ids.git
    cd network-security-ids

> Start with Docker Compose
    
    docker compose up --build

FastAPI documentation is available at http://127.0.0.1:8000/docs
Streamlit interface is available at http://127.0.0.1:8501

> To run locally without Docker, start the FastAPI backend using

    uvicorn main:app --reload --port 8000

> In another terminal, run the Streamlit frontend using

    streamlit run streamlit_app.py

# File Descriptions

> main.py contains the FastAPI backend with prediction endpoints.
> streamlit_app.py provides the Streamlit interface for single and batch predictions.
> models/ contains the trained model and features.json.
> .github/workflows/ contains the CI/CD pipeline configuration.
> Dockerfile and docker-compose.yml provide the containerization setup.

# Example Usage

For single prediction, send a JSON payload to POST /predict. The response returns whether the flow is DDoS or BENIGN.

For batch prediction, upload a CSV file to POST /predict_batch_file. The response includes predictions for each row.

# Future Work

Future extensions could expand classification to multiple attack categories in the CICIDS2017 dataset. Additional feature engineering and advanced models such as XGBoost or deep learning could be explored. Deployment to cloud infrastructure with real-time data streaming and integration with dashboards and alerting systems would enhance operational usability.

# License

This project was developed for academic and research purposes as part of a Master’s thesis at Dublin Business School.