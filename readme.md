# House Price Prediction

A full-stack machine-learning application that estimates Bangalore house prices from a property's location, total area, number of bedrooms, and number of bathrooms. The project includes a Python/FastAPI backend that serves predictions and a React/Vite frontend that provides the user interface.

## Model

The backend uses **Linear Regression** from scikit-learn. The model learns a relationship between the input features and the house price in Indian lakhs:

```text
price = intercept
      + (total_sqft × area coefficient)
      + (bath × bathroom coefficient)
      + (bhk × bedroom coefficient)
      + location contribution
```

Locations are converted into one-hot encoded features. Locations with fewer than 30 records are grouped into `other`, which is represented by the baseline when the model is trained.

### Training and preprocessing

The training script:

1. Loads the Bangalore dataset.
2. Removes unused columns and rows with missing values.
3. Converts values such as `2 BHK` and area ranges such as `1200-1400` into numeric values.
4. Removes area, price-per-square-foot, BHK, and bathroom outliers.
5. One-hot encodes the location.
6. Splits the data into training and test sets.
7. Trains and evaluates a `LinearRegression` model.
8. Saves the model and its metadata in `backend/artifacts/`.

The saved metadata contains the model features, supported locations, coefficients, intercept, R² score, and mean squared error.

## Datasets

The `Datasets/` directory contains:

- `Bangalore  house data.csv` - the dataset currently used by `backend/train.py`.
- `Delhi house data.csv` - an additional dataset available for future expansion.
- `Pune house data.csv` - an additional dataset available for future expansion.
- `data_cleaning_and_modeling.ipynb` - exploratory data cleaning and modeling work.
- `model.ipynb` - notebook-based model experimentation.

The current API and frontend are configured for Bangalore locations and prices. The Delhi and Pune datasets are not included in the current training pipeline.

## Folder structure

```text
house_price_prediction/
├── Datasets/
│   ├── Bangalore  house data.csv
│   ├── Delhi house data.csv
│   ├── Pune house data.csv
│   └── *.ipynb
├── backend/
│   ├── artifacts/
│   │   ├── model.joblib
│   │   └── model_metadata.json
│   ├── src/
│   │   ├── main.py
│   │   └── schemas.py
│   ├── train.py
│   ├── pyproject.toml
│   └── uv.lock
├── frontend/
│   ├── src/
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
└── readme.md
```

## Requirements

- Python 3.12 or newer
- `uv`
- Node.js and npm

## How to run

### 1. Train the model (optional)

The repository includes trained artifacts. To retrain the model after changing the data or training code:

```bash
cd backend
uv sync
uv run python train.py
```

This updates:

- `backend/artifacts/model.joblib`
- `backend/artifacts/model_metadata.json`

### 2. Start the backend

From the `backend/` directory:

```bash
uv run uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at `http://127.0.0.1:8000`.

Useful endpoints:

- `GET /api/model-info` - returns model status and available locations.
- `POST /api/predict` - returns a predicted price for the submitted property details.

### 3. Start the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal, usually `http://localhost:5174`.

The Vite development server proxies `/api` requests to the backend at `http://127.0.0.1:8000`.

## How the project can be enhanced

- **Improve model quality:** Compare Linear Regression with Ridge, Lasso, Random Forest, Gradient Boosting, or XGBoost using cross-validation.
- **Add stronger validation:** Report MAE, and R², and use a separate validation set for more reliable evaluation.
- **Support multiple cities:** Create a shared preprocessing pipeline and let users select Bangalore, Delhi, or Pune.
- **Use a saved pipeline:** Save preprocessing and the model together with scikit-learn `Pipeline` and `ColumnTransformer` objects to prevent training and prediction feature mismatches.
- **Improve data cleaning:** Handle inconsistent location names, unusual area formats, duplicate records, and extreme prices more systematically.
- **Add more useful features:** Include furnishing status, floor number, total floors, property type, age, parking, and amenities when reliable data is available.
- **Add prediction explanations:** Display the feature contributions already calculated by the API in a visual breakdown.
- **Productionize the application:** Add automated tests, structured logging, environment-based configuration, Docker deployment, monitoring, and a production web server.
- **Improve the frontend:** Add input validation messages, loading states with a progress indicator, charts, recent predictions, and accessible keyboard-friendly controls.

## Notes

If the model artifacts are unavailable, the backend serves a clearly marked mock prediction so the frontend can still be tested. For real predictions, train the model or keep the provided artifacts in `backend/artifacts/`.
