
FROM masroorhasan/pyspark:latest


RUN pip install findspark

COPY am3329_test.py /data/am3329_test.py
COPY ValidationDataset.csv /data/ValidationDataset.csv
COPY trainingmodel.model /data/trainingmodel.model
WORKDIR /data

ENTRYPOINT ["python", "am3329_test.py"]
CMD ["ValidationDataset.csv"]


