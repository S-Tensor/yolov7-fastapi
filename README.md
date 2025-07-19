### 1. Clone the repo

git clone https://github.com/your-username/yolov7-fastapi.git
cd yolov7-fastapi

### 2. Install dependencies
cd yolov7  
pip install -r requirements.txt

### 3 Run Code
uvicorn main:app --reload  
Visit: http://127.0.0.1:8000/docs for Swagger UI

### 4. API Endpoint
POST /detect  

Request Body:  
{
  "image_base64": "BASE64_IMAGE_STRING"
}
Response:

{
  "detected_objects": ["person", "dog", "car"]
}


