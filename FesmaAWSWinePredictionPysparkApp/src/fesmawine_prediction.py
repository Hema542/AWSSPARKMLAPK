import sys
from typing import List

from pyexpat import model
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.dataframe import DataFrame


def clean_data(df: DataFrame) -> DataFrame:
    """Cleans the DataFrame by removing extra quotes and casting to double."""
    return df.select(*(col(c).cast("double").alias(c.strip("\"")) for c in df.columns))


def train_and_tune(
    spark: SparkSession,
    train_data: DataFrame,
    valid_data: DataFrame,
    feature_cols: List[str],
    model_path: str,
) -> None:
    """Trains and tunes a RandomForestClassifier for wine quality prediction."""

    # Feature Assembler
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # Label Indexer
    indexer = StringIndexer(inputCol="quality", outputCol="label")

    # Random Forest Classifier
    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=150,
        maxBins=8,
        maxDepth=15,
        seed=150,
        impurity="gini",
    )

    # Pipeline
    pipeline = Pipeline(stages=[assembler, indexer, rf])

    # Initial Model Training and Evaluation
    model = pipeline.fit(train_data)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )  # Using F1 score for evaluation
    initial_f1 = evaluator.evaluate(model.transform(valid_data))
    print("Initial model F1 score:", initial_f1)

    # Hyperparameter Tuning
    paramGrid = (
        ParamGridBuilder()
        .addGrid(rf.maxBins, [8, 16, 32])  # Exploring different binning strategies
        .addGrid(rf.maxDepth, [10, 15, 20])  # Trying different tree depths
        .addGrid(rf.numTrees, [100, 200, 300])  # Varying the number of trees
        .addGrid(rf.featureSubsetStrategy, ["auto", "sqrt", "log2"])  # Feature selection strategies
        .addGrid(rf.impurity, ["gini", "entropy"])  # Trying different impurity measures
        .build()
    )

    crossval = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3,  # Using 3-fold cross-validation
    )
    cv_model = crossval.fit(train_data)

    # Best Model Evaluation
    best_model = cv_model.bestModel
    best_f1 = evaluator.evaluate(best_model.transform(valid_data))
    print("Best model F1 score:", best_f1)

    # Save the Best Model
    best_model.write().overwrite().save(model_path)


if __name__ == "__main__":
    print(f"Arguments: {sys.argv}")
    # Check for required arguments
    if len(sys.argv) > 3:
        input_path = sys.argv[1]
        valid_path = sys.argv[2]
        output_path = sys.argv[3] + "testmodel.model"
    else:
        input_path = "data/csv/TrainingDataset.csv"  # Default paths
        valid_path = "data/csv/ValidationDataset.csv"
        output_path = "data/model/testmodel.model"


    # Create SparkSession
    spark = SparkSession.builder.appName("fesma_wine_prediction").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Input/Output paths (now with error handling)
    input_path = sys.argv[1]
    valid_path = sys.argv[2]
    output_path = sys.argv[3] + "testmodel.model"

    # Read datasets
    train_data = clean_data(
        spark.read.format("csv")
        .option("header", "true")
        .option("sep", ";")
        .option("inferschema", "true")
        .load(input_path)
    )
    valid_data = clean_data(
        spark.read.format("csv")
        .option("header", "true")
        .option("sep", ";")
        .option("inferschema", "true")
        .load(valid_path)
    )

    # Feature columns
    feature_cols = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]

    # Train and tune the model
    train_and_tune(spark, train_data, valid_data, feature_cols, output_path)