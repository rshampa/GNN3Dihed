#!/bin/bash

file_base=("molrep" "train")

for base in "${file_base[@]}"; do
    file_inp="${base}.py"
    file_log="${base}.log"
    file_log_runtime="runtime_${base}.log"

    echo "Running $file_inp ...."
    python $file_inp > $file_log 2> $file_log_runtime

    if [ $? -ne 0 ]; then
        echo "$file_inp failed. Exiting."
        exit 1
    fi

    echo "$file_inp completed successfully."
done

echo "Both jobs completed successfully."
