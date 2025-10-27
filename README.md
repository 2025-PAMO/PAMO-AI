# PAMO-AI


## 🚀 로컬 실행 가이드 (프론트 테스트용)

### 1. Python 가상환경 설정
```bash
# 가상환경 생성 (Python 3.10 기준)
python3 -m venv venv

# 가상환경 활성화 (Mac / Linux)
source venv/bin/activate

# 가상환경 활성화 (Windows)
venv\Scripts\activate

```

### 2. 의존성 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

### 3. FastAPI 서버 실행

```bash
# 기본 실행 (자동 리로드)
uvicorn backend_python.main:app --reload --host 0.0.0.0 --port 8000

```

> Note:
> 
> - 프론트에서 API 요청 시 `.env` 또는 설정 파일에서 `API_BASE_URL`을 `http://localhost:8001`로 맞추세요.
> - 백엔드(Spring)와 포트가 겹치지 않도록 8001 포트를 사용합니다.

### 4. 가상환경 종료

```bash
deactivate

```

---

이렇게 하면 프론트 쪽에서 **`npm start`**로 띄운 뒤,

백엔드(Spring)와 FastAPI 둘 다 동시에 실행 가능해서 테스트 편하게 할 수 있습니다.
