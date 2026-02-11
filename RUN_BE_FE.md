# Chạy Backend và Frontend

## Chạy bằng Docker Compose (khuyến nghị)

Ở thư mục gốc project:

```bash
docker compose up -d --build
```

- Backend: http://localhost:1234  
- Frontend: http://localhost:3003  

Dừng: `docker compose down`

---

## Chạy tay (không Docker)

### Backend (Python - port 1234)

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 1234
```

Hoặc: `python app.py` (nếu có `if __name__ == "__main__"`).

## Frontend (Node.js - port 3003)

```bash
cd frontend
npm install
npm start
```

Mở trình duyệt: **http://localhost:3003**

Lưu ý: Cần chạy backend (1234) trước khi dùng chức năng load model và phân loại ảnh trên frontend.
