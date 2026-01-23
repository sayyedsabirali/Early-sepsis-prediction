FROM python:3.9-slim

WORKDIR /app

# copy only requirements first (cache-friendly)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# now copy project files
COPY . .

EXPOSE 5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]