"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function ApacheSpark() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          When your dataset outgrows a single machine — hundreds of gigabytes to petabytes — you need
          <strong> distributed computing</strong>. Apache Spark is the dominant engine for large-scale data
          processing. It lets you write code that <em>looks</em> like it runs on one machine but actually
          executes across a cluster of hundreds or thousands of nodes.
        </p>
        <p>
          Spark&apos;s core abstraction is the <strong>Resilient Distributed Dataset (RDD)</strong> — an
          immutable, partitioned collection of records spread across the cluster. You apply
          <strong> transformations</strong> (map, filter, groupBy) that are lazy (nothing runs until you
          trigger an <strong>action</strong> like collect, count, or save). This lets Spark optimize the
          entire computation graph before executing it.
        </p>
        <p>
          In practice, most people use <strong>DataFrames</strong> (the higher-level API built on top of
          RDDs). DataFrames look like pandas tables but are distributed. Spark&apos;s Catalyst optimizer
          compiles DataFrame operations into efficient physical execution plans, often beating hand-written
          RDD code. <strong>PySpark</strong> is the Python API for Spark, making it accessible to data
          scientists who already know pandas.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>MapReduce Paradigm</h3>
        <p>
          Spark generalizes the MapReduce model. Given a dataset <InlineMath math="D = \{d_1, d_2, \ldots, d_n\}" /> partitioned across <InlineMath math="P" /> nodes:
        </p>
        <BlockMath math="\text{Map: } f: d_i \rightarrow (k_i, v_i) \quad \text{(per-record transformation)}" />
        <BlockMath math="\text{Reduce: } g: (k, [v_1, v_2, \ldots]) \rightarrow (k, v_{\text{agg}}) \quad \text{(aggregation by key)}" />

        <h3>Data Partitioning</h3>
        <p>
          For a dataset of size <InlineMath math="N" /> with <InlineMath math="P" /> partitions, each
          partition has approximately <InlineMath math="N/P" /> records. The optimal number of partitions
          is typically:
        </p>
        <BlockMath math="P_{\text{optimal}} \approx 2\text{-}4 \times \text{(total CPU cores across cluster)}" />

        <h3>Shuffle Cost</h3>
        <p>
          Operations like <code>groupBy</code> and <code>join</code> require data redistribution (a &quot;shuffle&quot;).
          The network cost of a shuffle is:
        </p>
        <BlockMath math="\text{Shuffle cost} = O\left(\frac{N}{P} \cdot P\right) = O(N)" />
        <p>
          In practice, shuffles are the dominant bottleneck because they involve disk I/O, serialization,
          network transfer, and deserialization. Minimizing shuffles is the key to Spark performance.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>PySpark Basics</h3>
        <CodeBlock
          language="python"
          title="pyspark_basics.py"
          code={`from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Create a Spark session
spark = SparkSession.builder \\
    .appName("NeuronAcademy") \\
    .master("local[*]") \\
    .config("spark.sql.shuffle.partitions", 200) \\
    .getOrCreate()

# ---- Read data ----
df = spark.read.parquet("s3a://my-bucket/events/")
print(f"Rows: {df.count()}, Partitions: {df.rdd.getNumPartitions()}")

# ---- DataFrame transformations (lazy!) ----
result = (
    df
    .filter(F.col("event_type") == "purchase")           # WHERE
    .withColumn("revenue", F.col("price") * F.col("qty"))  # New column
    .groupBy("category")                                    # GROUP BY
    .agg(
        F.sum("revenue").alias("total_revenue"),
        F.count("*").alias("num_transactions"),
        F.avg("revenue").alias("avg_revenue"),
    )
    .orderBy(F.desc("total_revenue"))                       # ORDER BY
)

# Nothing has executed yet! Spark built an execution plan.
# This triggers execution:
result.show(10)

# ---- Write results ----
result.write.mode("overwrite").parquet("s3a://my-bucket/aggregated/")`}
        />

        <h3>RDD Operations</h3>
        <CodeBlock
          language="python"
          title="rdd_operations.py"
          code={`# ---- RDD: Lower-level API ----
sc = spark.sparkContext

# Create an RDD from a list
rdd = sc.parallelize(range(1_000_000), numSlices=100)

# Transformations (lazy)
squared = rdd.map(lambda x: x ** 2)
evens = squared.filter(lambda x: x % 2 == 0)

# Action (triggers execution)
total = evens.reduce(lambda a, b: a + b)
print(f"Sum of squared even numbers: {total}")

# ---- Word count (the classic MapReduce example) ----
text_rdd = sc.textFile("hdfs:///data/books/*.txt")

word_counts = (
    text_rdd
    .flatMap(lambda line: line.lower().split())     # Split lines into words
    .map(lambda word: (word, 1))                    # Map each word to (word, 1)
    .reduceByKey(lambda a, b: a + b)                # Sum counts by word
    .sortBy(lambda x: -x[1])                        # Sort by count descending
)

top_20 = word_counts.take(20)
for word, count in top_20:
    print(f"{word}: {count}")`}
        />

        <h3>Joins and Window Functions</h3>
        <CodeBlock
          language="python"
          title="spark_joins_windows.py"
          code={`from pyspark.sql.window import Window

# ---- Broadcast join (small table joins large table) ----
users = spark.read.parquet("users/")         # 1M rows (small)
events = spark.read.parquet("events/")       # 1B rows (large)

# Broadcast the small table to all nodes (avoids shuffle!)
from pyspark.sql.functions import broadcast
joined = events.join(broadcast(users), on="user_id", how="left")

# ---- Window functions ----
# Rank users by total spend within each region
window_spec = Window.partitionBy("region").orderBy(F.desc("total_spend"))

ranked = (
    joined
    .groupBy("user_id", "region")
    .agg(F.sum("revenue").alias("total_spend"))
    .withColumn("rank", F.row_number().over(window_spec))
    .filter(F.col("rank") <= 10)   # Top 10 per region
)

ranked.show()

# ---- Repartition for performance ----
# Before a join on user_id, co-partition both datasets
events_partitioned = events.repartition(200, "user_id")
users_partitioned = users.repartition(200, "user_id")
# Now the join is partition-local (no shuffle!)`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Use DataFrames, not RDDs</strong>: The Catalyst optimizer generates far better execution plans than hand-written RDD code. Only drop to RDDs for custom partitioning or complex stateful operations.</li>
          <li><strong>Broadcast small tables</strong>: If one side of a join fits in memory (&lt;10GB by default), use <code>broadcast()</code> to avoid a full shuffle. This can speed joins by 10-100x.</li>
          <li><strong>Partition wisely</strong>: Too few partitions underutilizes the cluster. Too many creates scheduling overhead. Aim for partitions of 128MB-256MB each.</li>
          <li><strong>Cache strategically</strong>: <code>df.cache()</code> stores intermediate results in memory. Use it when you read the same data multiple times. Call <code>df.unpersist()</code> when done.</li>
          <li><strong>Avoid collect()</strong>: <code>df.collect()</code> pulls all data to the driver node. On large datasets, this will crash the driver. Use <code>.show()</code>, <code>.take(n)</code>, or write to storage instead.</li>
          <li><strong>Spark UI is your best debugging tool</strong>: The web UI (port 4040) shows execution plans, stage timelines, shuffle sizes, and skew. Always check it when a job is slow.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Using Python UDFs when built-in functions exist</strong>: Python UDFs serialize data between JVM and Python, causing 10-100x slowdown. Always check if <code>pyspark.sql.functions</code> has what you need first.</li>
          <li><strong>Ignoring data skew</strong>: If one key has 10x more data than others, one partition does 10x more work. Fix with salting (append random suffix to keys) or pre-aggregation.</li>
          <li><strong>Not monitoring shuffle size</strong>: Large shuffles are the #1 performance killer. If your job shuffles 500GB, rethink your approach (broadcast join, pre-partitioning, or pre-aggregation).</li>
          <li><strong>Calling .count() to check if data exists</strong>: <code>.count()</code> scans the entire dataset. Use <code>.head(1)</code> or <code>.isEmpty()</code> instead.</li>
          <li><strong>Not setting spark.sql.shuffle.partitions</strong>: The default (200) is often wrong. Set it based on your data size: <code>total_data_size / 128MB</code>.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> You have a 10TB clickstream table and a 50MB user dimension table. How do you efficiently join them in Spark?</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Broadcast join</strong>: The user table (50MB) fits comfortably in memory. Use <code>broadcast(users)</code> so Spark sends the small table to every executor. This eliminates the shuffle of the 10TB table entirely.</li>
          <li><strong>Under the hood</strong>: Without broadcast, Spark would hash-partition both tables by the join key and shuffle 10TB across the network. With broadcast, only 50MB is sent to each node, and the join is done locally on each partition.</li>
          <li><strong>Verification</strong>: Check the physical plan (<code>df.explain()</code>) to confirm <code>BroadcastHashJoin</code> is used, not <code>SortMergeJoin</code>.</li>
          <li><strong>If the small table were 20GB</strong> (too large to broadcast): partition both tables by the join key (<code>repartition(N, &quot;user_id&quot;)</code>), then use a sort-merge join. Pre-bucketing the data at write time avoids the shuffle at join time.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>&quot;Learning Spark&quot; by Damji, Wenig, Das, Lee (2nd edition)</strong> — Comprehensive guide from Spark committers, covers Spark 3.x features.</li>
          <li><strong>Spark: The Definitive Guide by Chambers &amp; Zaharia</strong> — Detailed reference from the creator of Spark.</li>
          <li><strong>Databricks blog on Adaptive Query Execution</strong> — How Spark 3.x dynamically optimizes joins and shuffle partitions at runtime.</li>
          <li><strong>Jacek Laskowski&apos;s &quot;The Internals of Apache Spark&quot;</strong> — Deep dive into Catalyst, Tungsten, and execution engine internals.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
