We provide an example of the trained model on the MP e form task. 

1. The model use JARVIS as D1 in step1 and MP as D2 in step2.
Limited by the maximum file size of 50 Mb and the anonymity, 
we only provide one trained model(CranbNet on 100% mp e form dataset).


2. How to test our model
2.1 We also provide the CranbNet network file(kingcrab.py) and the test dataset
to prove the test MAE of the T-S CranbNet. You can use
***
xz -d mp_e_form_MPdata_JVpretrain_MPfinetune_ratio1.0.pth.xz
***
to unzip the checkpoint file.

2.2 Because we want to keep the code and hyperparameters true to the original 
version, you can copy the residual code from https://github.com/anthony-wang/CrabNet
to test our pretrained model(Thanks to the architecture-agnostic nature of our strategy).


3. All code, related data, and trained models will be available on github 
upon publication of the paper.