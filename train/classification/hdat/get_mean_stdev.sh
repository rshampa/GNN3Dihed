#!/bin/bash

filename="train"
grep -A 1 "Test Results:" $filename.log | awk '{print $3}' > temp_rmse.txt

# Run the Python script to compute mean and standard deviation
python - <<END
import numpy as np
import statistics

with open('temp_rmse.txt', 'r') as file:
    values = [line.strip() for line in file if line.strip()]
    values = [float(value) for value in values]

values_array = np.array(values)

mean = statistics.mean(values_array)
std_dev = statistics.stdev(values_array)

print(f"Mean: {mean:.3f} | Standard Deviation: {std_dev:.3f}")
END

# Clean up temporary file
rm temp_rmse.txt

