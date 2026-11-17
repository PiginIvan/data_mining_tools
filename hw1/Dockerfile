FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY hw1.py match_timestamps.cpp ./
RUN g++ -O3 -std=c++11 -shared -fPIC \
    -o libmatch_timestamps.so match_timestamps.cpp

CMD ["python", "hw1.py"]
