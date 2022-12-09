
FROM datamechanics/spark:3.1-latest

ENV PYSPARK_MAJOR_PYTHON_VERSION=3

WORKDIR /wrkdr
RUN pip3 install -r findspark

COPY /am3329_test.py .
COPY ValidationDataset.csv .

ENTRYPOINT ["python", "am3329_test.py", "ValidationDataset.csv"]

