# TempoKGAT Repository

This repository contains the implementation of **TempoKGAT**, a novel Graph Attention Network designed for temporal graph analysis, published in ICONIP2024. TempoKGAT combines time-decaying weights and selective neighbor aggregation to uncover latent patterns in spatio-temporal data. The model is evaluated on datasets from traffic, energy, and health sectors and demonstrates superior predictive accuracy over state-of-the-art methods.

## Features
- Full implementation of the TempoKGAT model.
- Scripts for dataset preparation, model training, and evaluation.
- Preprocessed datasets used in the experiments.

## Reference
For further details, please refer to the paper:  
**"[TempoKGAT: A Novel Graph Attention Network Approach for Temporal Graph Analysis"](https://arxiv.org/abs/2408.16391)**  
by Lena Sasal, Daniel Busby, and Abdenour Hadid.


## Requirements
------------
- Python 3.x
- PyTorch
- PyTorch Geometric Temporal
- Matplotlib

## Setup and Installation
----------------------
1. Ensure Python 3 is installed on your system. You can download it from https://www.python.org/downloads/.

2. Clone this repo and Install the required Python packages. You can use the following command to install all necessary libraries from the provided requirements.txt file:
```bash
git clone https://github.com/CapWidow/TempoKGATCode.git

pip install -r requirements.txt
```

Adjust the versions according to your compatibility needs.

## Running the Program
-------------------
1. Navigate to the directory containing the script `tempoKGATScript.py`.

2. Run the script using the following command line syntax. You can adjust the parameters according to your needs:

  -h, --help         show this help message and exit
  --dataset DATASET  Dataset to use (PedalMe, Chickenpox, EnglandCovid, WindmillSmall, WindmillMedium)
  --k K              Number of neighbors for topk selection
  --epochs EPOCHS    Number of epochs to train the model

Example:
```bash
python tempoKGATScript.py --dataset WindmillMedium --k 17 --epochs 200
```

3. The program will execute and save the plots to the specified directory (default : /results/), outputting the training and testing results in the console.

## Note:
-----
- Make sure SSL certificates are up to date if you encounter any SSL errors during dataset fetching. This script includes a workaround for SSL verification issues commonly encountered on MacOS and Windows.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Cite

If this research help you in your work please cite [TempoKGAT](https://arxiv.org/abs/2408.16391) :

```
@misc{sasal2024tempokgatnovelgraphattention,
      title={TempoKGAT: A Novel Graph Attention Network Approach for Temporal Graph Analysis}, 
      author={Lena Sasal and Daniel Busby and Abdenour Hadid},
      year={2024},
      eprint={2408.16391},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.16391}, 
}
```
