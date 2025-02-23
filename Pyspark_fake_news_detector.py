# Import necessary modules from PySpark
from pyspark.sql import SparkSession                   # To create the Spark session
from pyspark.sql.functions import col, lower, regexp_replace, trim  # For DataFrame column operations
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF  # For text feature extraction
from pyspark.ml import Pipeline                        # To build an ML pipeline

# Initialize the Spark session with an application name
spark = SparkSession.builder.appName("FakeNewsDetection").getOrCreate()

# ----------------------------------------------------------------------------------
# 1. Load the Fake News Dataset
# ----------------------------------------------------------------------------------
# Define the path to the CSV file containing the dataset.
# Replace 'path_to_dataset/news.csv' with the actual path to your dataset.
data_path = "path_to_dataset/news.csv"

# Read the CSV file into a DataFrame.
# 'header' option tells Spark that the first row is a header,
# 'inferSchema' automatically detects the data types of each column.
df = spark.read.option("header", "true").option("inferSchema", "true").csv(data_path)

# Print the schema of the DataFrame to verify that the columns are loaded correctly.
df.printSchema()

# ----------------------------------------------------------------------------------
# 2. Data Processing: Clean the Text Data
# ----------------------------------------------------------------------------------
# Create a new column 'clean_text' by performing the following steps:
# a. Convert the text to lowercase.
# b. Remove non-alphabetical characters using a regular expression.
# c. Trim any extra whitespace from the text.
df_clean = df.withColumn("clean_text", lower(col("text"))) \
             .withColumn("clean_text", regexp_replace(col("clean_text"), "[^a-zA-Z\\s]", "")) \
             .withColumn("clean_text", trim(col("clean_text")))

# ----------------------------------------------------------------------------------
# 3. Feature Extraction: Build a Text Processing Pipeline
# ----------------------------------------------------------------------------------
# Define the pipeline stages:

# a. Tokenizer: Splits the cleaned text into individual words.
tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")

# b. StopWordsRemover: Removes common stop words (e.g., 'the', 'is', etc.) from the list of words.
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

# c. HashingTF: Converts the filtered words into fixed-length feature vectors.
#    'numFeatures' defines the size of the feature vector.
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)

# d. IDF: Computes the Inverse Document Frequency, which scales the features based on their importance.
idf = IDF(inputCol="raw_features", outputCol="features")

# Build the Pipeline by combining all the stages.
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])

# Fit the pipeline on the cleaned DataFrame. This step learns the IDF weights.
model = pipeline.fit(df_clean)

# Transform the DataFrame using the fitted pipeline to generate feature vectors.
df_features = model.transform(df_clean)

# Select the columns of interest: cleaned text, extracted features, and the label column.
df_final = df_features.select("clean_text", "features", "label")

# Display the first 5 rows of the final DataFrame without truncating the text.
df_final.show(5, truncate=False)

# Stop the Spark session to free up resources.
spark.stop()
