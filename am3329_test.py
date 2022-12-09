import findspark
findspark.init()
findspark.find()
# Import essential functions
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,isnan, when, count
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession\
        .builder\
        .appName("test_model")\
        .getOrCreate()
spark.sparkContext.setLogLevel("Error")

# Load CSV file
#s3://am3329-cc648-wineapp/ValidationDataset.csv
df = spark.read.format("csv").load(sys.argv[1], header=True, inferschema=True, sep=';')

# Convert float to double
select_expr = [
    col(c).cast("float") if t == "double" else col(c) for c, t in df.dtypes
]
df = df.select(*select_expr)

# list of independent features
features_train = df.columns
features_train.remove('""""quality"""""')
df = df.withColumnRenamed('""""quality"""""', 'label')

# Vectorize for pyspark
vector_assembler = VectorAssembler(inputCols=features_train, outputCol = 'features')
df = vector_assembler.transform(df)

# Standardize the features
scaler = StandardScaler(inputCol = 'features', outputCol='scaled_features', withStd=True, withMean=True)
scaler = scaler.fit(df)
df = scaler.transform(df)

# Model Prediction
model_1 = LogisticRegressionModel.load("s3://am3329-cc648-wineapp/model/trainmodel.model/")
pred_train_lr = model_1.transform(df)
#pred_train_lr = pred_train_lr.withColumnRenamed('""""quality"""""', 'label')
evaluator = MulticlassClassificationEvaluator(metricName="f1")
print("F1 Score for Logistics Regression Model: ", evaluator.evaluate(pred_train_lr))

# # model_2 = DecisionTreeClassifierModel.load(sys.argv[2])
# # pred_train_dt = model_2.transform(df)
# # #pred_train_dt = pred_train_dt.withColumnRenamed('""""quality"""""', 'label')
# # evaluator_dt = MulticlassClassificationEvaluator(metricName="f1")
# # f1score_dt = evaluator_dt.evaluate(pred_train_dt)
# # print("F1 Score for Decision Tree Classifier Model: ", f1score_dt)

# model_3 = RandomForestModel.load(sys.argv[2])
# pred_train_rf = model_3.transform(df)
# #pred_train_rf = pred_train_rf.withColumnRenamed('""""quality"""""', 'label')
# evaluator_rf = MulticlassClassificationEvaluator(metricName="f1")
# print("F1 Score for Random Forest Classifier Model: ", evaluator_rf.evaluate(pred_train_rf))
