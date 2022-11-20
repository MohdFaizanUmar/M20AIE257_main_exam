FROM python:3.8.1
WORKDIR /app

COPY . /app

RUN pip3 --no-cache-dir install -r requirements.txt

CMD ["python","main_Q3.py","-optional_flag"]
