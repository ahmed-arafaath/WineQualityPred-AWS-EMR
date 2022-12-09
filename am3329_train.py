#import of essential functions
import findspark
findspark.init()
findspark.find()
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,isnan, when, count
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier

spark = SparkSession\
        .builder\
        .appName("training_model")\
        .getOrCreate()
spark.sparkContext.setLogLevel("Error")

# load CSV file
#s3://am3329-cc648-wineapp/TrainingDataset.csv
df = spark.read.format("csv").load(sys.argv[1], header=True, inferschema=True, sep=';')
df = df.distinct()
print(df.printSchema())

# Find Count of Null, None, NaN of All DataFrame Columns
print(df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]
   ).show())

# Convert float to double
select_expr = [
    col(c).cast("float") if t == "double" else col(c) for c, t in df.dtypes
]
df = df.select(*select_expr)

# list of independent features
features_train = df.columns
features_train.remove('""""quality"""""')

# Vectorize for pyspark
vector_assembler = VectorAssembler(inputCols=features_train, outputCol = 'features')
df = vector_assembler.transform(df)

# Standardize the features
scaler = StandardScaler(inputCol = 'features', outputCol='scaled_features', withStd=True, withMean=True)
scaler = scaler.fit(df)
df = scaler.transform(df)

# train test split and a list for f1 score tracking
df = df.withColumnRenamed('""""quality"""""', 'label')
train, test = df.randomSplit([0.8, 0.2], seed=42)
f1score = []

# Model Training
model_1 = LogisticRegression(featuresCol='scaled_features',labelCol='label')
model_1 = model_1.fit(train)
pred_train_lr = model_1.transform(test)
#pred_train_lr = pred_train_lr.withColumnRenamed('""""quality"""""', 'label')
evaluator = MulticlassClassificationEvaluator(metricName="f1")
f1score_lr = evaluator.evaluate(pred_train_lr)
print("F1 Score for Logistics Regression Model: ", f1score_lr)
f1score.append(f1score_lr)

model_2 = DecisionTreeClassifier(labelCol='label', featuresCol='scaled_features', maxDepth=25, 
                                minInstancesPerNode=30)
model_2 = model_2.fit(train)
pred_train_dt = model_2.transform(test)
#pred_train_dt = pred_train_dt.withColumnRenamed('""""quality"""""', 'label')
evaluator_dt = MulticlassClassificationEvaluator(metricName="f1")
f1score_dt = evaluator_dt.evaluate(pred_train_dt)
print("F1 Score for Decision Tree Classifier Model: ", f1score_dt)
f1score.append(f1score_dt)

model_3 = RandomForestClassifier(labelCol='label', featuresCol='scaled_features', maxDepth=25, 
                                numTrees=30)
model_3 = model_3.fit(train)
pred_train_rf = model_3.transform(test)
#pred_train_rf = pred_train_rf.withColumnRenamed('""""quality"""""', 'label')
evaluator_rf = MulticlassClassificationEvaluator(metricName="f1")
f1score_rf = evaluator_rf.evaluate(pred_train_rf)
print("F1 Score for Random Forest Classifier Model: ", f1score_rf)
f1score.append(f1score_rf)


# sameModel = LogisticRegressionModel.load(sc, "lrm_model.model")
# Save the model with highest f1 score
f1score.sort()
if f1score_lr == f1score[2]:
    model_1.write().overwrite().save('s3://am3329-cc648-wineapp/model/trainmodel.model')
    print("Of the three trained models, Logistic Regression gave the highest F1 score")
elif f1score_dt == f1score[2]:
    model_2.write().overwrite().save('s3://am3329-cc648-wineapp/model/trainmodel.model')
    print("Of the three trained models, Decision Tree Classifier gave the highest F1 score")
elif f1score_rf == f1score[2]:
    model_3.write().overwrite().save('s3://am3329-cc648-wineapp/model/trainmodel.model')
    print("Of the three trained models, Random Forest gave the highest F1 score")