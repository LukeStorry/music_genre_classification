#!/bin/bash
module add languages/anaconda2/5.0.1.tensorflow-1.6.0
srun -p gpu --gres=gpu:1 -t 0-02:00 --mem=15G --pty bash
