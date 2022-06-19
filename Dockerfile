FROM python:3.8

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8501 

COPY . /app

CMD ["cd", "data", "&&", "./script.sh" ]

ENTRYPOINT ["streamlit", "run"]

CMD ["main.py"]
