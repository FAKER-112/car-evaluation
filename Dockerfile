# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for XGBoost/Scikit-learn
# libgomp1 is often needed for XGBoost multithreading
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and the dataset
COPY train.py .
COPY predict.py .
COPY dataset/ dataset/

# TRICK: Run the training script *during* the build process.
# This generates 'model.pkl' inside the image so it's ready immediately.
RUN python train.py

# Expose port 5000 for the Flask API
EXPOSE 5000

# Define the command to run the application
CMD ["python", "predict.py"]