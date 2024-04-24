# GAN frameworks for CT tasks
## Model Training
To train models, run one of the files:
- `EIGAN.py`
- `EIGANPrior.py`
- `CycleGAN.py`
- `ExtendedCycleGAN.py`
- `ExtendedCycleGANStructureX.py`
- `ExtendedCycleGANStructureY.py`
- `ConditionalCycleGAN.py`

which trains the model under the respective framework (c.f. paper for detail).

## Model Evaluation
To evaluate trained models, specify the values for the variables below, then run `Python test.py`. 
- `path` specifies the path containing the csv file with the loss and reconstruction metrics during training.
- `net_ckp_ct` specifies the path to the trained model.
