FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8080
HEALTHCHECK CMD-SHELL python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT','8080') + '/_stcore/health')"
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT:-8080} --server.headless=true"]
