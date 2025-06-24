import duckdb

# Path to your CSV file
csv_path = "cleaned_sensor_data.csv"

# Query: Select distinct crop labels
query = f"""
    SELECT DISTINCT label
    FROM read_csv_auto('{csv_path}')
"""

# Run query
result = duckdb.query(query).to_df()

# Display result
print(result)
