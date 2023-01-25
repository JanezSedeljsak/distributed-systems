#!/bin/bash

module load CUDA/10.1.243-GCC-8.3.0
nvcc -o histogram.out histogram.cu

files=("500" "640" "800" "1000" "1024_640" "1200" "1680_1050" "1920_1080" "2560_1080" "2560_1440" "4k" "4kp")

for i in "${!files[@]}"; do

    if [ -f "out/${files[i]}.txt" ]; then
        rm "out/${files[i]}.txt"
    fi

    for j in {1..10}; do
        srun --reservation=fri --gpus=1 ./histogram.out "images/${files[i]}.jpg" "out/${files[i]}.jpg" | grep -v "^srun" >> "out/${files[i]}.txt";
    done;

    echo "Done $(($i + 1)) - ${files[i]}";
    
done
