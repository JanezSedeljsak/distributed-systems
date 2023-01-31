#!/bin/bash

module load CUDA/10.1.243-GCC-8.3.0
nvcc -o histogram.out histogram.cu

files=("500" "1024_640" "1920_1080" "2560_1440" "4kp")

for i in "${!files[@]}"; do

    if [ -f "chunks/${files[i]}.txt" ]; then
        rm "chunks/${files[i]}.txt"
    fi

    for j in {1..10}; do
        srun --reservation=fri --gpus=1 ./histogram.out "images/${files[i]}.jpg" "chunks/${files[i]}.jpg" "1" | grep -v "^srun" >> "chunks/${files[i]}.txt";
    done;

    echo "Done $(($i + 1)) - ${files[i]}";
    
done
