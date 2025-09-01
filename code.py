from sklearn.preprocessing import StandardScaler
data = pd.read_csv("nsl-kdd.csv")
X = data.drop("label", axis=1)
X_scaled = StandardScaler().fit_transform(X)


from sklearn.ensemble import IsolationForest
model = IsolationForest()
model.fit(X_scaled)
preds = model.predict(X_scaled)


from keras.models import Model
from keras.layers import Input, Dense

input_dim = X_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoder = Dense(32, activation='relu')(input_layer)
bottleneck = Dense(16, activation='relu')(encoder)
decoder = Dense(32, activation='relu')(bottleneck)
output_layer = Dense(input_dim, activation='linear')(decoder)

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=128)


from py2neo import Graph

graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
graph.run("CREATE (:Device {name: '192.168.1.2'})-[:SENT_TO]->(:Device {name: '10.0.0.3'})")


from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("StreamingIDS").getOrCreate()
df = spark.readStream.format("kafka").option("subscribe", "network-logs").load()

# Parse Kafka message to structured data


# Cybersecurity IDS Starter Codebase (Unsupervised + Real-time)

# 1. --- Data Preprocessing ---
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset (e.g., NSL-KDD or CICIDS2017)
data = pd.read_csv("nsl_kdd.csv")
X = data.drop(columns=["label", "attack_type"], errors='ignore')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. --- Isolation Forest Anomaly Detection ---
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(n_estimators=100, contamination=0.05)
iso_forest.fit(X_scaled)
iso_preds = iso_forest.predict(X_scaled)  # -1: anomaly, 1: normal

# 3. --- Autoencoder Anomaly Detection ---
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np

input_dim = X_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation="relu")(input_layer)
bottleneck = Dense(16, activation="relu")(encoded)
decoded = Dense(32, activation="relu")(bottleneck)
output_layer = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=128, shuffle=True)

reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
thresh = np.percentile(mse, 95)
autoencoder_preds = np.where(mse > thresh, -1, 1)

# 4. --- Neo4j Graph Insertion Example ---
from py2neo import Graph

graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))

# Example: Creating device nodes and connection
graph.run("""
    MERGE (a:Device {ip: '192.168.1.10'})
    MERGE (b:Device {ip: '10.0.0.3'})
    MERGE (a)-[:SENT_TO]->(b)
""")

# 5. --- Kafka Streaming Consumer Skeleton ---
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("IDSStream").getOrCreate()
df = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "network-logs") \
    .load()

# Add preprocessing logic and model inference here

df.writeStream \
    .outputMode("append") \
    .format("console") \
    .start() \
    .awaitTermination()

# 6. --- Evaluation Metrics ---
from sklearn.metrics import classification_report

y_true = data["label"] if "label" in data.columns else None
if y_true is not None:
    print("Isolation Forest:\n", classification_report(y_true, iso_preds))
    print("Autoencoder:\n", classification_report(y_true, autoencoder_preds))
