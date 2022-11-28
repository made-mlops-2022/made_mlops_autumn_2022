FROM tiangolo/uvicorn-gunicorn:python3.8

WORKDIR /app

COPY . /app
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# CMD ["uvicorn", "server:app"]
# CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]


