# Official Implementation for 'Unified Framework for Open-World Compositional Zero-shot Learning'

---
<p align="center">
  <img align="middle" src="network.png" alt="The main figure"/>
</p>

## Instalation
```
python=3.9.17
pytorch=1.13.1
```

## Dataset
To download datasets,
```
sh download_data.sh
```
To run the model for MIT-States Dataset:
Training:
```
python train.py with cfg=config/mit-states.yml per_gpu_batchsize=32 num_freeze_layers=0 lr_transformer=3.5e-6 lr=3.6e-6 lr_cross=1e-6 k=3 offset_val=0.1 neta=0.01

```
*This is a draft version of the final code. We will be cleaning up the code in following days.

