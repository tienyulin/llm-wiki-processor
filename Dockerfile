FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

EXPOSE 8001

CMD ["python", "main.py"]
