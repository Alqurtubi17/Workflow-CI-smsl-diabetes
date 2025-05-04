FROM python:3.9

WORKDIR /app

COPY MLProject/ /app

RUN pip install mlflow scikit-learn pandas matplotlib python-dotenv

ENTRYPOINT ["mlflow", "run", "."]