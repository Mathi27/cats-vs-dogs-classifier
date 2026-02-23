# Cats vs Dogs Classifier: End-to-End MLOps Pipeline

An production-grade End-to-End MLOps pipeline designed to classify images of cats and dogs. This project demonstrates the integration of deep learning with modern DevOps practices, including containerization, automated CI/CD, and real-time monitoring.

## Overview
The Cats vs Dogs Classifier is a full-lifecycle machine learning system. It utilizes a Convolutional Neural Network (CNN) to achieve high accuracy in image classification while ensuring the model is deployable, scalable, and monitorable in a production environment.

## Key Features
Automated Pipeline: Streamlined flow from data ingestion to model deployment.

- REST API: Flask/FastAPI based web interface for real-time image inference.
- Containerization: Fully Dockerized environment using docker-compose for multi-service orchestration.
- Monitoring: Integrated with Prometheus to track system metrics and model performance.
- CI/CD: Automated workflows via GitHub Actions for testing and deployment.
- Database Integration: Persistent storage for prediction logs and metadata.

# Tech Stack Overview

| Category                   | Tools                          |
|----------------------------|--------------------------------|
| Deep Learning              | TensorFlow, Keras              |
| Backend                    | Python, Flask / FastAPI        |
| Frontend                   | HTML5, Jinja2 Templates        |
| DevOps & Infrastructure    | Docker, Docker Compose         |
| Monitoring                 | Prometheus                     |
| CI/CD                      | GitHub Actions                 |
| Database                   | SQLite, PostgreSQL             |

---

If you want a **more professional README-style explanation version**, here’s an enhanced one:


# Project Structure

### 🔹 CI/CD
- `.github/workflows/` – GitHub Actions pipeline definitions for automated testing and deployment.

### 🔹 Application Layer
- `app.py` – Main Flask/FastAPI application entry point.
- `templates/` – Frontend HTML templates (Jinja2 based).

### 🔹 Database
- `db.py` – Database connection and management logic.

### 🔹 Testing
- `tests/` – Unit and integration test cases.

### 🔹 DevOps
- `dockerfile` – Docker image configuration.
- `docker-compose.yaml` – Multi-container orchestration (App + Monitoring).
- `prometheus.yml` – Prometheus monitoring configuration.

### 🔹 ML Model
- `best_model.h5` – Pre-trained CNN model file.

### 🔹 Dependencies & Config
- `requirements.txt` – Python package dependencies.
- `.env.example` – Sample environment variables configuration.

#  Pipeline Workflow

##  Data Processing
- Standardization of cat and dog image datasets  
- Image resizing and normalization  
- Data augmentation (rotation, flipping, zooming, etc.)  
- Train-validation split  

## Model Training
- Convolutional Neural Network (CNN) built using TensorFlow/Keras  
- Automated checkpointing to save best-performing model  
- Model evaluation on validation dataset  
- Final trained model saved as `best_model.h5`  

## Inference Service
- Web-based UI for image upload  
- Real-time prediction (Cat 🐱 / Dog 🐶)  
- Backend API handles preprocessing and inference  
- Returns classification result instantly  

## Monitoring
- Prometheus scrapes application metrics  
- Tracks:
  - Application health
  - Request count
  - Request latency
  - Error rate  

##  Automation (CI/CD)
- Every commit triggers GitHub Actions workflow  
- Runs:
  - Unit tests  
  - Integration tests  
  - Build verification  
- Ensures production-ready deployment

# Setup & Installation for Reproducability

## Prerequisites
- Python 3.9+
- Docker & Docker Compose

## Local Installation
```bash
git clone https://github.com/Mathi27/cats-vs-dogs-classifier.git
cd cats-vs-dogs-classifier
```

## Install dependencies:
```bash
pip install -r requirements.txt
```

## Run the app
```bash
python app.py
```

## Docker Deployment
Deploy the entire stack (App + Prometheus) using a single command:
```bash
docker-compose up --build
```

## Monitoring
Once the services are running via Docker:

Web App: Access at http://localhost:5000

Prometheus Metrics: Access at http://localhost:9090

## Testing
Run the automated test suite to ensure pipeline stability:
```bash
pytest tests/
```
