import os
import sys
from typing import Tuple

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import PipelineModel
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col


def clean_data(df: DataFrame) -> DataFrame:
    """Cleans the DataFrame by removing extra quotes and casting to double."""
    return df.select(*(col(c).cast("double").alias(c.strip("\"")) for c in df.columns))


def predict_and_evaluate(
    spark: SparkSession, test_data: DataFrame, model_path: str
) -> Tuple[float, float]:
    """Loads the trained model and evaluates its performance on test data."""

    # Load the trained model
    model = PipelineModel.load(model_path)

    # Make predictions
    predictions = model.transform(test_data)

    # Evaluate using F1 score and accuracy
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    f1_score = evaluator.evaluate(predictions)

    evaluator.setMetricName("accuracy")
    accuracy = evaluator.evaluate(predictions)

    # Calculate weighted F1 (using MLlib for convenience)
    results = predictions.select(["prediction", "label"])
    metrics = MulticlassMetrics(results.rdd.map(tuple))
    weighted_f1 = metrics.weightedFMeasure()

    return accuracy, weighted_f1


if __name__ == "__main__":
    print(f"Arguments: {sys.argv}")
    # Check for required argument
    if len(sys.argv) < 2:
        print("Error: Missing input argument. Please provide the test data path.")
        sys.exit(1)

    # Create SparkSession
    spark = SparkSession.builder.appName("fesma_wine_prediction").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Input/Output paths
    input_path = sys.argv[1]
    if not ("/" in input_path):
        input_path = f"data/csv/{input_path}"
    model_path = "/code/data/model/testdata.model"

    # Read test dataset
    test_data = clean_data(
        spark.read.format("csv")
        .option("header", "true")
        .option("sep", ";")
        .option("inferschema", "true")
        .load(input_path)
    )

    # Predict and evaluate
    accuracy, weighted_f1 = predict_and_evaluate(spark, test_data, model_path)

    print("Test Accuracy:", accuracy)
    print("Weighted F1 Score:", weighted_f1)