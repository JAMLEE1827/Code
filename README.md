# DiffuTraj
# Code

## Environment
    PyTorch == 1.7.1
    Python 3.8
    CUDA > 10.1

## Prepare Data

The preprocessed data are in ```raw_data```. We preprocess the data and generate .pkl files for training.

Run

```
python process_data.py
```

Data Format

```
Frame	MMSI	longitude(x)	latitude(y)
```

## Training

### Step 1: Modify or create your own config file in ```/configs``` 

```
./configs/YOUR_CONFIG.yaml
```

 ### Step 2: Train DiffuTraj

 ```python main.py --config configs/YOUR_CONFIG.yaml --dataset DATASET``` 

Logs and checkpoints will be automatically saved.

## Evaluation

To evaluate a trained-model, please set ```eval_mode``` in config file to True and set the epoch you'd like to evaluate at from ```eval_at``` and run

 ```python main.py --config configs/YOUR_CONFIG.yaml --dataset DATASET``` 

Since diffusion model is an iterative process, the evaluation process may take a little time (default 5 steps using DDIM).
