<h1 style="border-bottom: 2px solid lightgray;">TriPercept:  Bridging the Structural Perception Gap in Molecular Graphs</h1>

__Abstract:__ Accurate molecular property prediction requires a balanced perception of struc- tural information at multiple levels, including atomic attributes, topological patterns, and geometric configurations.  However, most existing models rely on a single struc- tural view, resulting in structural perception imbalance and poor generalization across tasks.   Given  the  inherent  complementarity  of these structural levels,  we introduce TriPercept, a unified representation learning framework designed to unify and integrate them.  TriPercept extracts atomic, topological, and geometric features via dedicated encoders, combines them through a structure-aware graph network that incorporates heterogeneous edge information into message passing, and reinforces structural con- sistency through contrastive self-supervised pretraining.  Evaluated on multiple clas- sification and regression tasks from MoleculeNet, TriPercept consistently outperforms competitive baselines on the majority of tasks, especially in structurally diverse sce- narios. Further experiments, including t-SNE visualizations of feature embeddings and atomic-level attention heatmaps, demonstrate the effectiveness of TriPercept’s multi- level structural fusion, highlighting its ability to focus on chemically relevant features and capture complex structural relationships.  

![](overall.png)

## Getting Started

### Installation

Set up conda environment and clone the github repo

```
cd ./TriPercept-main
conda env create -f requirement.yaml
conda activate TriP
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch_geometric==1.6.3
pip install torch_scatter==2.0.7
pip install torch_sparse==0.6.9
pip install azureml-defaults
pip install rdkit-pypi cython
python setup.py build_ext --inplace
python setup_cython.py build_ext --inplace
pip install -e .
pip install --upgrade protobuf==3.20.1
pip install --upgrade tensorboard==2.9.1
pip install --upgrade tensorboardX==2.5.1
```
###Pre-training

To train the TriPercept, where the configurations and detailed explaination for each variable can be found in config.yaml

```
$ python TriP.py
```

### Training

To finetune the TriPercept pre-trained model on downstream molecular benchmarks, where the configurations and detailed explaination for each variable can be found in config_finetune.yaml
```
$ python finetune.py
```





