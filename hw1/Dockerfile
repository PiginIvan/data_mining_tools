FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY hw1.py .

CMD ["python", "hw1.py"]
