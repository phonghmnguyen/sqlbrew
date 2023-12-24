FROM python:3.8.5

WORKDIR /app

COPY "./requirements.txt" .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm

COPY . .

EXPOSE 4000

CMD ["python", "server.py"]

