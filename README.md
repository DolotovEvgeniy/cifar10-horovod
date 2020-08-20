# Train neural network for image classification (CIFAR-10) with horovod

## Run Docker
1. [Install Docker](https://www.docker.com/get-started)
2. Build image:
```
docker build -t horovod:latest docker
```
3. Run container:
```
docker run -it horovod:latest
```

## Run training
```
horovodrun -np 3 python train.py --num_epochs=3 --batch-size=20 --lr=0.001 |& grep -v "Read -1"
```

## Experiments:
Accuracy after 3 epochs: 68%
