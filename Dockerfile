
FROM python:3.9

WORKDIR /home/reza/Code/dlmg/

COPY . .
RUN pip install -r requirements.txt

COPY scripts/ ./scripts/

CMD ["python", "./scripts/preprocess.py"]
