# Customer Churn Prediction and Model Deployment



A machine learning project that predicts customer churn using Logistic Regression and exposes predictions through a FastAPI REST API.



## Features



- Logistic Regression classification pipeline

- Single-customer and batch predictions

- API key authentication

- Data drift monitoring using Population Stability Index (PSI)

- Automated testing with Pytest

- Model metadata and version tracking

- Interactive Swagger API documentation



## Technology Stack



- Python

- Pandas and NumPy

- Scikit-learn

- FastAPI and Uvicorn

- Joblib

- Pytest



## Model Evaluation



| Metric | Result |

|---|---:|

| Cross-validation ROC-AUC | 0.8485 |

| Test ROC-AUC | 0.8423 |

| Accuracy | 0.6785 |

| Precision | 0.4470 |

| Recall | 0.8904 |

| F1-score | 0.5952 |

| Decision threshold | 0.17 |



These are results from the current development evaluation. The model's high recall helps identify potential churners, although some positive predictions may be incorrect.



## Project Structure



```text

Churn_Model_Deployment/

├── app/

│   └── main.py

├── artifacts/

│   ├── churn_pipeline.joblib

│   └── model_metadata.json

├── data/

│   ├── raw/

│   └── processed/

├── reports/

├── src/

│   ├── __init__.py

│   ├── drift_monitor.py

│   ├── preprocessing.py

│   └── train.py

├── tests/

│   └── test_api.py

├── drift_test.json

├── requirements.txt

├── .gitignore

└── README.md

```



## Installation



Create and activate a virtual environment on Windows:



```cmd

python -m venv .venv

.venv\\Scripts\\activate

```



Install the dependencies:



```cmd

pip install -r requirements.txt

```



## Run the API



Set your API key in Command Prompt:



```cmd

set CHURN_API_KEY=replace-with-your-own-secret

```



Start the server:



```cmd

uvicorn app.main:app --host 127.0.0.1 --port 8000

```



Open the interactive API documentation:



http://127.0.0.1:8000/docs



Health endpoint:



http://127.0.0.1:8000/health



## API Endpoints



| Method | Endpoint | Purpose |

|---|---|---|

| GET | `/health` | Check API health |

| POST | `/predict` | Predict churn for one customer |

| POST | `/predict/batch` | Predict churn for multiple customers |

| POST | `/monitor/drift` | Monitor feature distribution drift |



Prediction and drift endpoints require API key authentication. Refer to Swagger documentation for the exact request fields.



## Testing



Run the automated tests:



```cmd

set CHURN_API_KEY=ChurnProjectLocal2026

pytest -v

```



The current development test run passed 16 tests. Run the tests again after making code or dependency changes.



## Drift Monitoring



The application monitors feature distribution changes using PSI.



- Low drift: PSI below 0.10

- Moderate drift: PSI from 0.10 through 0.20

- High drift: PSI above 0.20



A previous test using records from the reference dataset reported low overall drift. This validates the monitoring workflow, but does not prove that future production data will remain stable.



## Security



- Configure API keys through environment variables.

- Never commit real secrets or `.env` files.

- Replace development credentials before deployment.

- Check dataset licensing and privacy requirements before publishing data.



## Limitations



Predictions are estimates, not guarantees of customer behavior. Model performance and drift should be evaluated using representative, authorized data before production use.



## Author



Nivetha S.



Machine Learning | FastAPI | Model Deployment | Testing | Drift Monitoring



