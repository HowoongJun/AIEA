# Active Image Enhancing Agent (AIEA) for Keypoint Detection
This work includes deep reinforcement learning based method that can adaptively enhance images to find keypoint better.
The code is implemnted with pytorch.

## Environment


## Usage
To run this code, type command below with the arguments. 

```bash
$ python run.py --mode <MODE> --db <DB> --query <QUERY>
```

Argument | Meaning | Contents
---|:---:|---:
`--mode`, `-o`| Train or Evaluation mode. | train, eval
`--db`, `-d`| DB folder for training (Only used in the train mode)| DB_FOLDER_PATH
`--query`, `-q`| Query image folder for evaluation (Only used in the eval mode)|QUERY_FOLDER_PATH

