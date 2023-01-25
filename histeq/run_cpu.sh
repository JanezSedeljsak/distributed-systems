#!/bin/bash

gcc histogram.c -lm -o histogram_cpu.out 

files=("500" "640" "800" "1000" "1024_640" "1200" "1680_1050" "1920_1080" "2560_1080" "2560_1440" "4k" "4kp")

for i in "${!files[@]}"; do

    if [ -f "out_cpu/${files[i]}.txt" ]; then
        rm "out_cpu/${files[i]}.txt"
    fi

    for j in {1..10}; do
        srun --reservation=fri -n1 --cpus-per-task=1 ./histogram_cpu.out "images/${files[i]}.jpg" "out_cpu/${files[i]}.jpg" | grep -v "^srun" >> "out_cpu/${files[i]}.txt";
    done;

    echo "Done $(($i + 1)) - ${files[i]}";
    
done

