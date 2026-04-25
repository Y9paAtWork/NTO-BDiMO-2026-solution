FROM python:3.12
WORKDIR /app


ADD requirements.txt .
RUN pip install -r requirements.txt
COPY . .

ENTRYPOINT python3 main.py
