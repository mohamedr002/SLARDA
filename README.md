# SLARDA

### A- Download the dataset from this following links
    - MFD: https://mb.uni-paderborn.de/en/kat/main-research/datacenter/bearing-datacenter/data-sets-and-download
    - HAR: https://archive.ics.uci.edu/ml/datasets/OPPORTUNITY+Activity+Recognitio
    - SSC:https://sleepdata.org/datasets/shhs, https://physionet.org/content/sleep-edf/1.0.0/
### B-  Add dataset to the framework
1- Add the data files in the following format:

    - Each domain splitted to train, val, test files
    - name the domains using small letters, i.e., a, b, c,...
    - Each sample has the following dict format: 
        - samples = data['samples']
        - labels = data['labels']
### Running the model 

1- from args:
    - Select the domain adaptation method as 'SLARDA'
    - Select your target dataset: Paderborn_FD, HAR, EEG
    - Select the corresponding based model: CNN_SL_bn, CNN_Opp_HAR_SL, EEG_M_SL

2- Run 'train_CD.py' script

