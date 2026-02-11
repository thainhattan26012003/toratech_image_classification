
FROM pytorch/pytorch:2.1.0-cpu AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart Pillow

COPY app.py .

EXPOSE 1234

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "1234"]
