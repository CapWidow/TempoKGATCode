README.txt
==========

Requirements
------------
- Python 3.x
- PyTorch
- PyTorch Geometric Temporal
- Matplotlib

Setup and Installation
----------------------
1. Ensure Python 3 is installed on your system. You can download it from https://www.python.org/downloads/.

2. Install the required Python packages. You can use the following command to install all necessary libraries from the provided requirements.txt file:

pip install -r requirements.txt


Adjust the versions according to your compatibility needs.

Running the Program
-------------------
1. Navigate to the directory containing the script `tempoKGATScript.py`.

2. Run the script using the following command line syntax. You can adjust the parameters according to your needs:

  -h, --help         show this help message and exit
  --dataset DATASET  Dataset to use (PedalMe, Chickenpox, EnglandCovid, WindmillSmall, WindmillMedium)
  --k K              Number of neighbors for topk selection
  --epochs EPOCHS    Number of epochs to train the model

Example:

python tempoKGATScript.py --dataset WindmillMedium --k 17 --epochs 200

3. The program will execute and save the plots to the specified directory (default : /results/), outputting the training and testing results in the console.

Note:
-----
- Make sure SSL certificates are up to date if you encounter any SSL errors during dataset fetching. This script includes a workaround for SSL verification issues commonly encountered on MacOS and Windows.