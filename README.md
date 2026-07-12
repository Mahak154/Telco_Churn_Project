# Customer Churn Prediction System

An end-to-end Data Science / ML / Cloud project: predicts which customers are
likely to churn, exposes the model as a REST API, and provides a Streamlit
dashboard for non-technical business users.

## Project Structure
```
churn_project/
├── data/
│   ├── generate_data.py          # synthetic Telco-style dataset generator
│   ├── telco_churn.csv           # raw dataset
│   └── telco_churn_processed.csv # cleaned + feature-engineered dataset
├── notebooks/
│   ├── eda.py                    # exploratory data analysis
│   ├── preprocess.py             # cleaning + feature engineering
│   ├── train_models.py           # model training, tuning, evaluation
│   └── test_predict.py           # sanity-check script for the model
├── models/
│   ├── churn_model.joblib        # final tuned Gradient Boosting model
│   ├── label_encoders.joblib
│   ├── scaler.joblib
│   └── feature_columns.joblib
├── app/
│   ├── main.py                   # FastAPI prediction service
│   ├── dashboard.py               # Streamlit dashboard
│   └── requirements.txt
├── outputs/                       # all charts generated for the report
├── Dockerfile                     # containerizes the API
├── Dockerfile.dashboard           # containerizes the dashboard
└── docker-compose.yml             # runs both together
```

## Running Locally

```bash
pip install -r app/requirements.txt

# Run the API
uvicorn app.main:app --reload --port 8000
# Visit http://localhost:8000/docs for interactive Swagger UI

# Run the dashboard (separate terminal)
streamlit run app/dashboard.py
# Visit http://localhost:8501
```

## Running with Docker

```bash
docker compose up --build
# API        -> http://localhost:8000
# Dashboard  -> http://localhost:8501
```

## Cloud Deployment (AWS)

The project is designed to deploy the same way it would in a real company setting:

1. **Model & artifact storage (S3)**
   Upload `models/*.joblib` to an S3 bucket (e.g. `s3://churn-prediction-artifacts/`)
   so the API can pull the latest model version at startup instead of bundling
   it into the Docker image — this makes retraining/redeployment cleaner.

2. **Containerize** — build the Docker image and push it to **Amazon ECR**:
   ```bash
   aws ecr create-repository --repository-name churn-api
   docker build -t churn-api .
   docker tag churn-api:latest <account_id>.dkr.ecr.<region>.amazonaws.com/churn-api:latest
   docker push <account_id>.dkr.ecr.<region>.amazonaws.com/churn-api:latest
   ```

3. **Deploy** — two common options:
   - **EC2**: launch an instance, install Docker, `docker run` the image, open port 8000 in the security group.
   - **Lambda + API Gateway** (serverless): wrap the FastAPI app with `Mangum` and deploy as a Lambda function behind API Gateway — cheaper for low/sporadic traffic.

4. **SageMaker alternative**: for a more "managed ML" story in the report, the
   same model can be registered as a SageMaker endpoint instead of a
   hand-rolled FastAPI service — worth mentioning as an alternative approach
   considered.

5. **Monitoring**: CloudWatch logs for the EC2/Lambda service; log every
   prediction request/response pair (input features, predicted probability,
   latency) to track model drift over time.

> Note: for a lighter-weight deployment (e.g. if AWS access is limited during
> the internship), the same Docker image deploys directly to **Render**,
> **Railway**, or **Streamlit Community Cloud** with no code changes — these
> are legitimate cloud platforms and a reasonable substitute to describe in
> the report if AWS credentials aren't available.

## Model Performance (Final: Tuned Gradient Boosting)

| Metric | Score |
|---|---|
| Accuracy | 0.768 |
| Precision | 0.678 |
| Recall | 0.492 |
| F1 Score | 0.570 |
| ROC-AUC | 0.834 |

Top churn drivers (feature importance): **Contract type**, **Monthly Charges**,
**Tenure**, **Tech Support**, **Online Security**.
