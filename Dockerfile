FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN mkdir -p /etc/ssl && \
    echo "[global]\ncert = /dev/null" > /root/.pip/pip.conf && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

EXPOSE 8001

CMD ["python", "main.py"]
