FROM python:3.8
COPY requirements.txt .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y 

RUN pip install  -r requirements.txt  --no-cache-dir &&  \
	rm requirements.txt 


RUN pip install python-multipart
EXPOSE 80
COPY ./app /app
WORKDIR /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
