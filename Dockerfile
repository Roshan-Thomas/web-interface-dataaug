FROM python:3.8
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8501
COPY . /app
# RUN cd /app/scripts && ./aravec_download.sh
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]