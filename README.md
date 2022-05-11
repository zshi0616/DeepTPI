## DeepTPI: Test Point Insertion with Deep Reinforcement Learning

Code repository for the paper:  
** DeepTPI: Test Point Insertion with Deep Reinforcement Learning **  
[Zhengyuan Shi](https://cure-lab.github.io/people/zhengyuan_shi/index.html), [Min Li](https://cure-lab.github.io/people/min_li/index.html), [Sadaf Khan](https://khan-sadaf.github.io/), Liuzheng Wang, Naixing Wang, Yu Huang and [Qiang Xu](https://cure-lab.github.io/qiang_xu.html)

# Abstract
Test point insertion (TPI) is a widely used technique for testability enhancement, especially for logic built-in self-test (LBIST) due to its relatively low fault coverage. In this paper, we propose a novel TPI approach based on deep reinforcement learning (DRL), named \emph{DeepTPI}. Unlike previous learning-based solutions that formulate the TPI task as a supervised-learning problem, we train a novel DRL agent with the Deep Q-learning algorithm. Specifically, we model circuits as directed graphs and embed a graph neural networks (GNNs) into the value network to predict the action value. Meanwhile, we leverage the general node embedding from a pre-trained model as a partial node feature and design a dedicated testability-aware attention mechanism for the value network. The ablation studies prove that our agent can learn a better policy with the above two methods. Experimental results on circuits with different scale show that DeepTPI significantly improves test coverage compared to existing solutions.

<!-- ## TPI for LBIST
The pseudo-random test patterns in LBIST are generated on chip to verify whether the correctness of circuit response. However, there are still some random pattern resistant (RPR) faults, whose patterns are very difficultly excited by random pattern generator. In order to detect these RPR faults and improve test coverage (TC), DFT engineers have to insert some extra gates into the netlist following the test point insertion (TPI) methods. These extra gates allow directly modifying the value somewhere inside the circuit and are named as control points (CP). 

## How does it work -->



# Installation
The experiments are conducted on Linux, with Python version 3.7.4, PyTorch version 1.8.1, and [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric) version 2.0.1.

To set up the environment:
```sh
conda create -n deepgate python=3.7.4
conda activate deepgate
pip install -r requirements.txt
```


# Running training code
To train the RL Value Network (Graph-DQN),
```sh
bash run/ITC22/train.sh
```
For settings of experiments, run the scripts in directory `./exp`.

# Running testing code
To test the RL agent,
```sh
bash run/ITC22/test.sh
```

