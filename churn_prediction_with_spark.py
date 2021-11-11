
################################### Churn Prediction with PySpark ###################################

################### Libraries and Utilities #####################

import warnings
import findspark
import pandas as pd
import seaborn as sns
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.ml.feature import Bucketizer
from pyspark.sql.functions import when, count, col

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

#### Findspark Modul
findspark.init("c:/spark")

#### Spark Session
spark = SparkSession.builder \
    .master("local") \
    .appName("pyspark_giris") \
    .getOrCreate()

sc = spark.sparkContext

################################ Exploratory Data Analysis ###############################

spark_df = spark.read.csv("datasets/churn2.csv", header=True, inferSchema=True)

def check_spark_df(dataframe: object) -> object:
    print("##################### Shape #####################")
    print((dataframe.count(), len(spark_df.columns)))

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.show(5, truncate=True))

    print("##################### Tail #####################")
    print(dataframe.tail(5))

    print("##################### NA #####################")
    print(dataframe.select([count(when(col(c).isNull(), c)).alias(c) for c in spark_df.columns]).toPandas().T)

    print("##################### Quantiles #####################")
    print(dataframe.describe().show())

check_spark_df(spark_df)

# Statistical Analysis
spark_df.describe(["Age", "Exited"]).show()

spark_df.select("NumOfProducts").distinct().show()

spark_df.groupby("Exited").count().show()

spark_df.groupby(["Exited", "Gender"]).agg({"CreditScore": "mean"}).show()

spark_df.groupby(["Exited", "NumOfProducts"]).agg({"Tenure": "mean"}).show()

spark_df.groupby("Exited").agg({"HasCrCard": "count"}).show()

spark_df.groupby("Exited").agg({"IsActiveMember": "count"}).show()

spark_df.groupby(["Exited", "NumOfProducts"]).agg({"EstimatedSalary": "mean"}).show()

# Selection of observations in the specified range
spark_df.filter(spark_df.Age > 40).count()
spark_df.filter(spark_df.Exited == 0).count()

# Selection and summary statistics of all numeric variables
num_cols = [col[0] for col in spark_df.dtypes if col[1] != 'string']

spark_df.select(num_cols).describe().toPandas().transpose()

# Selection and summary of all categorical variables
cat_cols = [col[0] for col in spark_df.dtypes if col[1] == 'string']

# Analysis of the Target Variable with Categorical Variables
for col in cat_cols:
    spark_df.select(col).distinct().show()

# Analysis of the Target Variable with Numerical Variables
for col in [col.lower() for col in num_cols]:
    spark_df.groupby("Exited").agg({col: "mean"}).show()


############################## Data Preprocessing & Feature Engineering #############################

##################### Feature Interaction ###############

spark_df = spark_df.withColumn('new_crdtscore_salary', spark_df.CreditScore / spark_df.EstimatedSalary)
spark_df = spark_df.withColumn('new_crdtscore_tenure', spark_df.CreditScore / spark_df.Tenure)
spark_df = spark_df.withColumn('new_tenure_age', spark_df.Tenure / spark_df.Age)

################ Bucketization / Bining / Num to Cat ############

# Age
spark_df.select('Age').describe().toPandas().transpose()
spark_df.select("Age").summary("count", "min", "10%", "25%", "50%", "75%", "90%", "95%", "99%", "max").show()

bucketizer = Bucketizer(splits=[0, 35, 55, 75, 95], inputCol="Age", outputCol="new_age_cat")
spark_df = bucketizer.setHandleInvalid("keep").transform(spark_df)
spark_df = spark_df.withColumn('new_age_cat', spark_df.new_age_cat + 1)  # labellar 0'dan başlasığı için +1 ekledim.

spark_df.groupby("new_age_cat").count().show()
spark_df.groupby("new_age_cat").agg({'Exited': "mean"}).show()
spark_df.groupby(["Exited", "new_age_cat"]).agg({'EstimatedSalary': "mean"}).show()
spark_df = spark_df.withColumn("new_age_cat", spark_df["new_age_cat"].cast("integer"))
spark_df.groupby("new_age_cat").agg({'Exited': "mean"}).show()

# EstimatedSalary
spark_df.select('EstimatedSalary').describe().toPandas().transpose()
spark_df.select("EstimatedSalary").summary("count", "min", "10%", "25%", "50%", "75%", "90%", "95%", "99%", "max").show()

bucketizer = Bucketizer(splits=[0, 50000, 100000, 150000, 200000], inputCol="EstimatedSalary", outputCol="new_estimatedsalary_cat")
spark_df = bucketizer.setHandleInvalid("keep").transform(spark_df)
spark_df = spark_df.withColumn('new_estimatedsalary_cat', spark_df.new_estimatedsalary_cat + 1)  #

spark_df.groupby("new_estimatedsalary_cat").agg({'Exited': "mean"}).show()

# Tenure
spark_df.select('Tenure').describe().toPandas().transpose()
spark_df.select("Tenure").summary("count", "min", "10%", "25%", "50%", "75%", "90%", "95%", "99%", "max").show()

bucketizer = Bucketizer(splits=[0, 3, 5, 7, 10], inputCol="Tenure", outputCol="new_tenure_cat")
spark_df = bucketizer.setHandleInvalid("keep").transform(spark_df)
spark_df = spark_df.withColumn('new_tenure_cat', spark_df.new_tenure_cat + 1)  #  I added +1 because labels start from 0.


############################# Label Encoding ############################

spark_df.show(5)
indexer = StringIndexer(inputCol="Gender", outputCol="gender_label")
indexer.fit(spark_df).transform(spark_df).show(5)

temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("gender_label", temp_sdf["gender_label"].cast("integer"))
spark_df.show(5)

spark_df = spark_df.drop('Gender')

############################# One Hot Encoding ##########################

encoder = OneHotEncoder(inputCols=["new_age_cat", "new_estimatedsalary_cat", "new_tenure_cat", "Gender"],
                        outputCols=["new_age_cat_ohe",  "new_estimatedsalary_cat_ohe", "new_tenure_cat_ohe", "gender_ohe"])

spark_df = encoder.fit(spark_df).transform(spark_df)
spark_df.show(5)

#############################  TARGET ############################

stringIndexer = StringIndexer(inputCol='Exited', outputCol='label')
temp_sdf = stringIndexer.fit(spark_df).transform(spark_df)

spark_df = temp_sdf.withColumn("label", temp_sdf["label"].cast("integer"))
spark_df.show(5)

############################# Defining Features ########################

features = ['CreditScore',
 'Age',
 'Tenure',
 'Balance',
 'NumOfProducts',
 'HasCrCard',
 'IsActiveMember',
 'EstimatedSalary',
 'Exited',
 'new_crdtscore_salary',
 'new_crdtscore_tenure',
 'new_tenure_age',
 'new_age_cat_ohe',
 'new_estimatedsalary_cat_ohe',
 'new_tenure_cat_ohe',
 'label']


# Vectorize independent variables.
va = VectorAssembler(inputCols=features, outputCol="features")
va_df = va.transform(spark_df)
va_df.show()

# Final sdf
final_df = va_df.select("features", "label")
final_df.show(5)

# StandardScaler
# scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
# final_df = scaler.fit(final_df).transform(final_df)

# Split the dataset into test and train sets.
train_df, test_df = final_df.randomSplit([0.7, 0.3], seed=17)
train_df.show(10)
test_df.show(10)

print("Training Dataset Count: " + str(train_df.count()))
print("Test Dataset Count: " + str(test_df.count()))


########################################## Modeling ###########################################

############################# Logistic Regression ############################


log_model = LogisticRegression(featuresCol='features', labelCol='label').fit(train_df)
y_pred = log_model.transform(test_df)
y_pred.show()

y_pred.select("label", "prediction").show()

# accuracy
y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()

evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName='areaUnderROC')
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

acc = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "accuracy"})
precision = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "precisionByLabel"})
recall = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "recallByLabel"})
f1 = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "f1"})
roc_auc = evaluator.evaluate(y_pred)

print("accuracy: %f, precision: %f, recall: %f, f1: %f, roc_auc: %f" % (acc, precision, recall, f1, roc_auc))

############################# Gradient Boosted Tree Classifier ############################

gbm = GBTClassifier(maxIter=100, featuresCol="features", labelCol="label")
gbm_model = gbm.fit(train_df)
y_pred = gbm_model.transform(test_df)
y_pred.show(5)

y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()

############################# Model Tuning ############################

evaluator = BinaryClassificationEvaluator()

gbm_params = (ParamGridBuilder()
              .addGrid(gbm.maxDepth, [2, 4, 6])
              .addGrid(gbm.maxBins, [20, 30])
              .addGrid(gbm.maxIter, [10, 20])
              .build())

cv = CrossValidator(estimator=gbm,
                    estimatorParamMaps=gbm_params,
                    evaluator=evaluator,
                    numFolds=5)

cv_model = cv.fit(train_df)


y_pred = cv_model.transform(test_df)
ac = y_pred.select("label", "prediction")
ac.filter(ac.label == ac.prediction).count() / ac.count()