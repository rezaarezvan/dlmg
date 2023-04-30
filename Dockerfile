FROM tensorflow/tensorflow:latest-gpu

WORKDIR /home/reza/Code/dlmg/

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "train.py"]
