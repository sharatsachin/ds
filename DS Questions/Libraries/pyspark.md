# PySpark

## What is PySpark and how does it differ from pandas?

PySpark is the Python API for Apache Spark, designed for big data processing. Unlike pandas which operates on a single machine, PySpark:
- Distributes data processing across multiple machines
- Handles data larger than memory
- Uses lazy evaluation
- Provides fault tolerance
- Optimized for parallel processing

## What are the core concepts in PySpark?

The core concepts are:
1. **SparkContext**: The main entry point for Spark functionality
2. **SparkSession**: The unified entry point of a Spark application (preferred starting Spark 2.0)
3. **RDD (Resilient Distributed Dataset)**: Basic abstraction for distributed data
4. **DataFrame**: Distributed collection of data organized into named columns
5. **Dataset**: Strongly-typed version of DataFrame (primarily used in Scala)

## How do you create a SparkSession?

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("MyApp") \
    .master("local[*]") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
```

## What are the common ways to create a DataFrame in PySpark?

1. From a list:
```python
data = [("John", 30), ("Alice", 25)]
df = spark.createDataFrame(data, ["name", "age"])
```

2. From a pandas DataFrame:
```python
pandas_df = pd.DataFrame({"name": ["John", "Alice"], "age": [30, 25]})
df = spark.createDataFrame(pandas_df)
```

3. From a CSV file:
```python
df = spark.read.csv("path/to/file.csv", header=True, inferSchema=True)
```

4. From a Parquet file:
```python
df = spark.read.parquet("path/to/file.parquet")
```

## What are the main DataFrame operations in PySpark?

1. **Select Operations**:
```python
# Select specific columns
df.select("name", "age")
# Select with expressions
df.select(col("age") + 1, col("name").alias("full_name"))
```

2. **Filter Operations**:
```python
# Filter with condition
df.filter(col("age") > 25)
# Multiple conditions
df.filter((col("age") > 25) & (col("name").startswith("J")))
```

3. **Grouping Operations**:
```python
# Group by and aggregate
df.groupBy("department").agg(avg("salary"), max("age"))
```

4. **Join Operations**:
```python
# Inner join
df1.join(df2, "employee_id", "inner")
# Left join with multiple conditions
df1.join(df2, ["dept_id", "location"], "left")
```

## What are the different types of joins available in PySpark?

PySpark supports these join types:
- **inner**: Returns only matching rows
- **outer** or **full** or **full_outer**: Returns all rows from both DataFrames
- **left** or **left_outer**: Returns all rows from left DataFrame
- **right** or **right_outer**: Returns all rows from right DataFrame
- **left_semi**: Returns rows from left DataFrame that have matches in right
- **left_anti**: Returns rows from left DataFrame that don't have matches in right
- **cross**: Returns Cartesian product of both DataFrames

## How do you handle missing values in PySpark?

Common operations for handling missing values:
```python
# Drop rows with any null values
df.na.drop()

# Drop rows where specific columns have null
df.na.drop(subset=["age", "salary"])

# Fill null values with a specific value
df.na.fill(0, subset=["age"])
df.na.fill({"age": 0, "salary": 50000})

# Replace null values using forward fill
window = Window.orderBy("date")
df.withColumn("value", last("value", True).over(window))
```

## What are the common aggregation functions in PySpark?

PySpark provides these built-in aggregation functions:
```python
from pyspark.sql.functions import sum, avg, min, max, count, countDistinct

df.agg(
    sum("salary").alias("total_salary"),
    avg("salary").alias("avg_salary"),
    min("salary").alias("min_salary"),
    max("salary").alias("max_salary"),
    count("employee_id").alias("employee_count"),
    countDistinct("department").alias("dept_count")
)
```

## How do you perform window functions in PySpark?

Window functions allow calculations across rows:
```python
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, rank, dense_rank, lag, lead

# Define window specification
windowSpec = Window.partitionBy("department").orderBy("salary")

# Add row number
df.withColumn("row_number", row_number().over(windowSpec))

# Add rank
df.withColumn("rank", rank().over(windowSpec))

# Add dense rank
df.withColumn("dense_rank", dense_rank().over(windowSpec))

# Add previous value
df.withColumn("prev_salary", lag("salary").over(windowSpec))

# Add next value
df.withColumn("next_salary", lead("salary").over(windowSpec))
```

## How do you handle date and timestamp operations in PySpark?

Common date operations:
```python
from pyspark.sql.functions import date_format, datediff, months_between, add_months

# Format date
df.withColumn("formatted_date", date_format("date", "yyyy-MM-dd"))

# Calculate date difference
df.withColumn("days_diff", datediff(col("end_date"), col("start_date")))

# Calculate months between
df.withColumn("months_diff", months_between(col("end_date"), col("start_date")))

# Add months
df.withColumn("future_date", add_months(col("date"), 3))
```

## How do you perform string operations in PySpark?

Common string functions:
```python
from pyspark.sql.functions import concat, substring, upper, lower, trim

# Concatenate strings
df.withColumn("full_name", concat(col("first_name"), lit(" "), col("last_name")))

# Substring
df.withColumn("short_name", substring(col("name"), 1, 3))

# Case conversion
df.withColumn("upper_name", upper(col("name")))
df.withColumn("lower_name", lower(col("name")))

# Trim whitespace
df.withColumn("trimmed_name", trim(col("name")))
```

## How do you save/write DataFrames in PySpark?

Common ways to save DataFrames:
```python
# Save as CSV
df.write.csv("path/to/output.csv", header=True)

# Save as Parquet
df.write.parquet("path/to/output.parquet")

# Save as table
df.write.saveAsTable("my_table")

# Save modes:
# - 'error' (default): error if exists
# - 'overwrite': overwrite if exists
# - 'append': append if exists
# - 'ignore': ignore if exists
df.write.mode("overwrite").parquet("path/to/output")
```

## How do you optimize PySpark performance?

Common optimization techniques:
1. **Caching**:
```python
# Cache DataFrame in memory
df.cache()
# Or specify storage level
df.persist(StorageLevel.MEMORY_AND_DISK)
```

2. **Partitioning**:
```python
# Repartition DataFrame
df.repartition(10)
# Repartition by specific column
df.repartition("department")
```

3. **Broadcast joins** for small DataFrames:
```python
from pyspark.sql.functions import broadcast
df1.join(broadcast(df2), "employee_id")
```

4. **Coalesce** to reduce partitions:
```python
df.coalesce(1)
```

## How do you handle UDFs (User Defined Functions) in PySpark?

Creating and using UDFs:
```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# Create UDF
@udf(returnType=StringType())
def upper_case(text):
    if text is None:
        return None
    return text.upper()

# Apply UDF
df.withColumn("upper_name", upper_case(col("name")))

# Alternative registration
spark.udf.register("upper_case_sql", upper_case)
# Use in SQL
spark.sql("SELECT upper_case_sql(name) FROM table")
```

## What are the common data types in PySpark?

PySpark SQL Types:
```python
from pyspark.sql.types import (
    StringType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    BooleanType,
    DateType,
    TimestampType,
    ArrayType,
    MapType,
    StructType,
    StructField
)

# Create schema
schema = StructType([
    StructField("name", StringType(), True),
    StructField("age", IntegerType(), False),
    StructField("scores", ArrayType(IntegerType()), True),
    StructField("properties", MapType(StringType(), StringType()), True)
])
```