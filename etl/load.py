import pandas as pd
import sqlite3

# Load dataset (adjust path after downloading from Kaggle)
df = pd.read_csv("./data/raw/diabetes_dataset.csv")

# Connect (or create) SQLite database
conn = sqlite3.connect("./database/health_indicators.db")

# Save dataframe to a table (replace if already exists)
df.to_sql("diabetes_health_indicators", conn, if_exists="replace", index=False)

# Verify by reading a few rows back
check = pd.read_sql("SELECT * FROM diabetes_health_indicators LIMIT 5;", conn)
print(check)

# Close connection
conn.close()
