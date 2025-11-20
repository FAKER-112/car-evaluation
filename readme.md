# Car Evaluation Classification Project

## Problem Description

The Car Evaluation Database provides a structured dataset used to predict the overall acceptability of a car based on six categorical attributes: buying price, maintenance cost, number of doors, passenger capacity, luggage boot size, and safety rating. The original data originates from a hierarchical decision model designed to evaluate cars through intermediate decision layers (PRICE, TECH, COMFORT). However, in this dataset, the hierarchical structure has been removed, leaving a direct mapping between the six input attributes and the final class label.

The predictive task is to build a classification model that accurately determines car acceptability into one of four possible classes: **unacceptable**, **acceptable**, **good**, or **very good**. No missing values exist, and all 1,728 possible attribute combinations are represented, making the attribute space complete. The class distribution is highly imbalanced, with the majority of instances labeled as unacceptable.

This dataset serves as a benchmark for machine learning methods, particularly those involving constructive induction, hierarchical model reconstruction, and classification under class imbalance. The problem requires developing and evaluating algorithms capable of learning meaningful decision boundaries within a fully discrete attribute space while handling skewed class distributions.

**Data Source:** [UCI Machine Learning Repository - Car Evaluation Dataset](https://archive.ics.uci.edu/dataset/19/car+evaluation)

## Project Structure

- **`dataset/`**: Contains the dataset file (`car.csv`).
- **`train.py`**: Script to train the XGBoost model. It loads data, preprocesses it (encoding), trains the model, evaluates it, and saves the trained model as `model.pkl`.
- **`app.py`**: A Streamlit web application that allows users to interactively input car features and get a prediction.
- **`predict.py`**: A Flask-based REST API that serves the trained model for predictions via HTTP requests.
- **`Dockerfile`**: Configuration to containerize the application.
- **`requirements.txt`**: List of Python dependencies.
- **`notebook.ipynb`**: Jupyter notebook for exploratory data analysis (EDA) and model experimentation.

## Setup and Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package installer)

### 1. Clone the Repository
```bash
git clone <repository_url>
cd <repository_directory>
```

### 2. Create and Activate a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## How to Run the Project

### 1. Train the Model
Before running the app or API, you need to train the model to generate the `model.pkl` file.

```bash
python train.py
```
This will output the classification report and save `model.pkl` in the project root.

### 2. Run the Streamlit Web App
To use the interactive web interface:

```bash
streamlit run app.py
```
This will open a browser window (usually at `http://localhost:8501`) where you can select car attributes and see the prediction.

### 3. Run the Flask API
To serve the model as a REST API:

```bash
python predict.py
```
The API will start on `http://localhost:5000`.

#### API Usage Example
You can send a POST request to the `/predict` endpoint.

**Using curl:**
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "buying": "vhigh",
           "maint": "med",
           "doors": "2",
           "persons": "2",
           "lug_boot": "small",
           "safety": "low"
         }'
```

**Response:**
```json
{
    "class_id": 0,
    "prediction": "Unacceptable"
}
```

## Docker Instructions

You can also run the project using Docker. The Dockerfile is configured to train the model during the build process (or you can modify it to use a pre-trained model) and then start the Flask API.

### 1. Build the Docker Image
```bash
docker build -t car-eval-app .
```

### 2. Run the Docker Container
```bash
docker run -p 5000:5000 car-eval-app
```

The API will now be accessible at `http://localhost:5000`.