FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .
COPY predict.py .
COPY data/ data/

# Run training to generate model.pkl and encoder.pkl
RUN python train.py

EXPOSE 5000

CMD ["python", "predict.py"]
