import pandas as pd

# Sample DataFrame with trip data
df = pd.DataFrame({
    'trip_id': ['T1', 'T1', 'T1', 'T2', 'T2'],
    'source': ['A', 'B', 'C', 'A', 'D'],
    'destination': ['B', 'C', 'D', 'B', 'E'],
    'actual_time': [10, 20, 40, 15, 5],
    'segment_time': [5, 10, 15, 5, 3]
})

# Step 1: Grouping the data by trip_id, source, and destination
# Aggregating the actual_time and segment_time for each group
grouped = df.groupby(['trip_id', 'source', 'destination']).agg({
    'actual_time': 'last',  # Get the last actual_time value for each trip
    'segment_time': 'sum'   # Get the sum of segment_times for each trip
}).reset_index()

print("Grouped Data:")
print(grouped)

# Step 2: Final aggregation at the trip_id level
# Aggregate by trip_id, get the last actual_time and the sum of segment_time
final_agg = grouped.groupby('trip_id').agg({
    'actual_time': 'last',   # Get the last actual_time for each trip
    'segment_time': 'sum'    # Get the sum of segment_time for each trip
}).reset_index()

print("\nFinal Aggregated Data:")
print(final_agg)
