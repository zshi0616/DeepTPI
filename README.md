## DeepTPI: Test Point Insertion with Deep Reinforcement Learning

Code repository for the paper:  
** DeepTPI: Test Point Insertion with Deep Reinforcement Learning **  
[Zhengyuan Shi](https://cure-lab.github.io/people/zhengyuan_shi/index.html), 
[Min Li](https://scholar.google.com/citations?user=X5gRH80AAAAJ&hl=zh-CN), 
[Sadaf Khan](https://khan-sadaf.github.io/), 
Liuzheng Wang, Naixing Wang, Yu Huang, and 
[Qiang Xu](https://cure-lab.github.io/qiang_xu.html)

# Abstract

## How does it work
In modern very large-scale integrated circuits, testing is a significant part to ensure the high-assurance and reliability. For example, logic build-in-self-test (LBIST) and Automatic test pattern generation (ATPG) technologies are used for manufacturing and reliability inspection. 

## TPI for LBIST and ATPG

The pseudo-random test patterns in LBIST are generated on chip to verify whether the correctness of circuit response. However, there are still some random pattern resistant (RPR) faults, whose patterns are very difficultly excited by random pattern generator. In order to detect these RPR faults and improve test coverage (TC), DFT engineers have to insert some extra gates into the netlist following the test point insertion (TPI) methods. These extra gates allow directly modifying the value somewhere inside the circuit and are named as control points (CP). 

However, different from LBIST, ATPG will generate determinant test cubes for every testable faults instead of relying on the random cubes. The test pattern is merged from test cubes, thus if there are some internal value conflict, more test patterns will be stored. To deal with this problem and further reduce pattern counts (PC), some extra CPs are inserted into circuits. Please refer [paper](https://ieeexplore.ieee.org/abstract/document/7342383) for more details about ATPG conflict. 

All in all, the objective of **TPI for LBIST** is **improving test converage** with random patterns and the objective of **TPI for ATPG** is **reducing pattern counts**. 

# RL Agent for LBIST task
｜ State | Netlist | 
| Action | Position and CP type (AND/OR) | 
| Reward | TC Improvement| 



# Installation
The experiments are conducted on Linux, with Python version 3.7.4, PyTorch version 1.8.1, and [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric) version 2.0.1.

To set up the environment:
```sh
git clone https://github.com/Ironprop-Stone/RL_TPI
cd RL_TPI
conda create -n deepgate python=3.7.4
conda activate deepgate
pip install -r requirements.txt
```

# Prepare dataset


# Running training code
To train the RL Value Network,
```sh
bash experiment/RL/train_agent.sh
```
For settings of experiments, run the scripts in directory `./exp`.

# Load trained model

# Run evaluation code

# Results


# DeepTPI
