# Deep Metric Learning: Caltech-101



### 1. Environment Setup
Ensure you have the required dependencies installed:
```bash
pip install -r requirements.txt 
```

2. *Dataset split:*
    Replace the path to you dataset inside split_dataset.py and execute the script. It will generate the train test val split. 
2. *Training:*
    Adjust the config.yaml file to specify the type of training to be used and run the train.py script
3. *Inference:*
    ```bash
    python inference.py --weights [path to the model weights] --image [path to the image]
    ```