# TruthFlow: Truthful LLM Generation via Representation Flow Correction

This repository contains the source code and instructions to reproduce the results in our paper, **[TruthFlow: Truthful LLM Generation via Representation Flow Correction](https://arxiv.org/pdf/2502.04556)**.


## Setup
Create an virtual python environment and use ``requirements.txt`` to set up all the required packages.
```bash
conda create -n TruthFlow python=3.11.9
conda activate TruthFlow
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requriements.txt
```
Please make sure that your device supports CUDA with 12.1 or higher version.



## Create Dataset For TruthFlow
To train and test TruthFlow, you have to first extract query last token hidden states and query-specific truthful direction.
Here is an example command to create dataset.
```bash
python create_ds.py --model_name gemma-2 --layers 18 20 22 --test_size 0.5 --seed 0 --token_pos ans_avg --ds_name tqa
```
### Explanation of the command
* ``--model_name`` Specifies the model.
* ``--layers`` Which layer(s) to extract hidden states.
* ``--test_size`` How to split dataset.
* ``--seed`` Set random seed to ensure of reproducibility.
* ``--token_pos`` How to average hidden states for truthful direction.
* ``--ds_name`` What dataset to use.

## Train and Test TruthFlow

After collecting training data, the TruthFlow is ready to train and test. The following command will run the training and evaluation process.
```bash
python flow.py --model_name gemma-2 --ds_path ./data_tqa/gemma-2_ans_avg_seed0_testsize0.5_layers_18_20_22 --layers 20 --seed 0 --method truthflow --opengen_eval --eval_method gpt --k 20 --alpha 1.5 --train --num_epochs 40

python flow.py --model_name gemma-2 --ds_path ./data_tqa/gemma-2_ans_avg_seed0_testsize0.5_layers_18_20_22 --layers 20 --seed 0 --method alphasteer --opengen_eval --eval_method gpt --k 20 --alpha 1.5 --train --num_epochs 40
```
### Explanation of the command
* ``--model_name`` Specifies the model.
* ``--layers`` Which layer to apply flow matching model. Should be only one layer!
* ``--seed`` Set random seed to ensure of reproducibility.
* ``--ds_path`` Local path to the data collected before for training and testing TruthFlow.
* ``--k`` How many top singular vectors to select to form the truthful subspace.
* ``--alpha`` The hyperparameter to control the intervention intensity. 
* ``--num_epochs`` How many epochs to train flow matching model.

## Acknowledgements

This work builds upon several open source projects. In particular:

- The implementation of our rectified flow model follows the construction design from **[rectified-flow-pytorch](https://github.com/lucidrains/rectified-flow-pytorch) by Phil Wang (lucidrains)**. We are grateful to the authors for making their excellent implementation publicly available.


## How to cite
```bibtex
@misc{wang2025truthflowtruthfulllmgeneration,
      title={TruthFlow: Truthful LLM Generation via Representation Flow Correction}, 
      author={Hanyu Wang and Bochuan Cao and Yuanpu Cao and Jinghui Chen},
      year={2025},
      eprint={2502.04556},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.04556}, 
}

```