# Exercise 7: image processing - image sharpening[^1]


## Goal:


The purpose of this exercise is to implement an image processing algorithm on the GPU, namely image sharpening


## Instructions:


- move to subdierctory:

```console

$ cd 07-image-filter/

```

- load CUDA module:

```console

$ module load CUDA

```

- build the application:

```console

$ make all

```


- Run the application on a compute node:

```console

$ srun --partition=gpu --ntasks=1 --gpus=1 prog helmet_in.png helmet_out.png

```


___


[^1]: 

 &copy; Patricio Bulić, University of Ljubljana, Faculty of computer and information science. 

 The materials are published under license [Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).


___
