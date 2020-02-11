#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 single.py --experiment mydataset --ewc --lambda=5000 --savepath=ewc
CUDA_VISIBLE_DEVICES=0 python3 single.py --experiment mydataset --ewc --online --lambda=5000 --gamma=1 --savepath=ewconline
CUDA_VISIBLE_DEVICES=0 python3 single.py --experiment mydataset --replay=current --distill --savepath=replay-current
CUDA_VISIBLE_DEVICES=0 python3 single.py --experiment mydataset --replay=generative --distill --savepath=replay-generative-distill
CUDA_VISIBLE_DEVICES=0 python3 single.py --experiment mydataset --xdg=0.8 --savepath=xdg
CUDA_VISIBLE_DEVICES=0 python3 single.py --experiment mydataset --si --c=0.1 --savepath=si
CUDA_VISIBLE_DEVICES=0 python3 single.py --experiment mydataset --replay=generative --savepath=replay-generative
CUDA_VISIBLE_DEVICES=0 python3 single.py --experiment mydataset --replay=generative --distill --feedback --savepath=replay-generative-distill-feedback
CUDA_VISIBLE_DEVICES=0 python3 single.py --experiment mydataset --savepath=naive
CUDA_VISIBLE_DEVICES=0 python3 single.py --experiment mydataset --savepath=culmulative --culmulative=1

