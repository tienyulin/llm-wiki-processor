FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
ENV PIP_TRUSTED_HOST="pypi.python.org pypi.org files.pythonhosted.org"
RUN pip install --no-cache-dir --trusted-host pypi.python.org --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

EXPOSE 8001

CMD ["python", "main.py"]
