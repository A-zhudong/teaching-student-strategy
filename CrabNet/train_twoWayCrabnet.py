import os
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score

from crabnet.kingcrab_prop_tru import CrabNet_twoWay
# from crabnet.kingcrab_transformer import CrabNet
from crabnet.model import Model
from utils.get_compute_device import get_compute_device

compute_device = get_compute_device(prefer_last=True)
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)


# %%
def get_model(data_dir, mat_prop, classification=False, batch_size=None,
              transfer=None, verbose=True, save_path='models/mp_bandgap_models/orimodels/',\
              model_name='mp_gappbe_embedding_loss', embedding_loss=False,\
              pretrained=False, finetune=False, embedding_path = None, transformer_model=False,\
              crab_ori_embedding=False):
    # Get the TorchedCrabNet architecture loaded
    print('model_name: ', model_name)
    model = Model(CrabNet_twoWay(compute_device=compute_device, pretrained=pretrained).to(compute_device),
                  model_name=model_name, verbose=verbose, embedding_loss=embedding_loss,\
                  save_path=save_path, pretrained=pretrained, finetune=finetune,\
                  embedding_path=embedding_path, transformer_model=transformer_model,\
                  crab_ori_embedding=crab_ori_embedding)

    # Train network starting at pretrained weights
    if transfer is not None:
        model.load_network(transfer)
        # model.model_name = f'{mat_prop}'

    # Apply BCEWithLogitsLoss to model output if binary classification is True
    if classification:
        model.classification = True

    # Get the datafiles you will learn from
    train_data = f'{data_dir}/{mat_prop}/train.csv'
    try:
        val_data = f'{data_dir}/{mat_prop}/val.csv'
    except:
        print('Please ensure you have train (train.csv) and validation data',
               f'(val.csv) in folder "data/materials_data/{mat_prop}"')

    # Load the train and validation data before fitting the network
    data_size = pd.read_csv(train_data).shape[0]
    batch_size = 2**round(np.log2(data_size)-4)
    if batch_size < 2**7:
        batch_size = 2**7
    if batch_size > 2**12:
        batch_size = 2**12
    batch_size = 512
    model.load_data(train_data, batch_size=batch_size, train=True)
    print(f'training with batchsize {model.batch_size} '
          f'(2**{np.log2(model.batch_size):0.3f})')
    model.load_data(val_data, batch_size=batch_size)

    # Set the number of epochs, decide if you want a loss curve to be plotted
    model.fit(epochs=300, losscurve=False)

    # Save the network (saved as f"{model_name}.pth")
    model.save_network()
    return model


def to_csv(output, save_name):
    # parse output and save to csv
    act, pred, formulae, uncertainty = output
    df = pd.DataFrame([formulae, act, pred, uncertainty]).T
    df.columns = ['composition', 'target', 'pred-0', 'uncertainty']
    save_path = 'model_predictions'
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(f'{save_path}/{save_name}', index_label='Index')


def load_model(data_dir, mat_prop, classification, file_name, verbose=True, model_name=None,\
                save_model_path=None, pretrained=False, crab_ori_embedding=False):
    # Load up a saved network.
    model = Model(CrabNet_twoWay(compute_device=compute_device, pretrained=pretrained).to(compute_device),\
                  model_name=model_name, verbose=verbose, save_path=save_model_path,\
                  crab_ori_embedding=crab_ori_embedding)
    model.load_network()
    # model.load_network('/home/zd/zd/teaching_net/CrabNet/models/pretrained_models/mp_e_form_pretrain_embedding.pth')

    # Check if classifcation task
    if classification:
        model.classification = True

    # Load the data you want to predict with
    data = f'{data_dir}/{mat_prop}/{file_name}'
    # data is reloaded to model.data_loader
    model.load_data(data, batch_size=2**9, train=False)
    return model


def get_results(model):
    output = model.predict(model.data_loader)  # predict the data saved here
    return model, output


def save_results(data_dir, mat_prop, classification, file_name, verbose=True,\
                 model_name=None, save_model_path=None, pretrained=False, crab_ori_embedding=False):
    model = load_model(data_dir, mat_prop, classification, file_name, verbose=verbose,\
                        model_name=model_name, save_model_path=save_model_path, pretrained=pretrained,
                        crab_ori_embedding=crab_ori_embedding)
    model, output = get_results(model)
    embeddings = output[4]
    # Get appropriate metrics for saving to csv
    if model.classification:
        auc = roc_auc_score(output[0], output[1])
        print(f'{mat_prop} ROC AUC: {auc:0.3f}')
    else:
        mae = np.abs(output[0] - output[1]).mean()
        print(f'{mat_prop} mae: {mae:0.3g}')

    # save predictions to a csv
    # fname = f'{mat_prop}_{file_name.replace(".csv", "")}_output.csv'
    # to_csv(output, fname)
    return model, mae, embeddings


# %%
if __name__ == '__main__':
    # Choose the directory where your data is stored
    # data_dir = 'data/materials_data'
    # data_dir = '/home/zd/zd/teaching_net/CrabNet/data/matbench_cv'
    # data_dir = '/home/zd/zd/teaching_net/alignn/alignn/data_formula_atom/mp_e_form/real_ma_e_form'
    # data_dir = '/home/zd/zd/teaching_net/alignn/alignn/data_formula_atom/'
    data_dir = '/home/zd/zd/teaching_net/data/jarvis'
    # Choose the folder with your materials properties
    # mat_prop = 'example_materials_property'
    # mat_prop = 'mp_e_form'
    prop = 'e_form'
    mat_prop = f'jarvis_{prop}'
    # save_model_name = 'mp_gappbe_pretrain_finetune_embloss'
    # save_model_path = 'models/mp_bandgap_models/pretrained_models/'
    save_model_name = f'jarvis_{prop}_twoway_epoch299_embloss_finetune'
    # save_model_name = f'jarvis_{prop}_struPred_epoch299_finetune'
    # save_model_path = f'/home/zd/zd/teaching_net/CrabNet/models/jarvis_{prop}_models/pretrained_models'
    save_model_path = f'/home/zd/zd/teaching_net/CrabNet/twoway_prop_stru/jarvis_{prop}/models'
    Transfer = None
    # Transfer = f'/home/zd/zd/teaching_net/CrabNet/models/jarvis_{prop}_models/pretrained_models/jarvis_{prop}_pretrain_epoch299.pth'  # the path of the pretrained model to be used
    # Transfer = f'/home/zd/zd/teaching_net/CrabNet/predict_structurEmbedding/jarvis_{prop}/models/jarvis_{prop}_struPred_epoch299.pth'
    embedding_loss = True  # whether to using the embedding to calculate the loss
    embedding_path = f'/home/zd/zd/teaching_net/data/jarvis/jarvis_{prop}/jarvis_eval_formula_embedding.pkl'
    # embedding_path = f'/home/zd/zd/teaching_net/CrabNet/predict_structurEmbedding/jarvis_{prop}/embeddings/jarvis_structureEmbedding.pkl'
    pretrain = True  # whether use the pretrained mode of the crab model structure
    finetune = True # whether to use the finetune mode to change the beta loss parameter
    crab_ori_embedding =False  # weather use the structure embedding prediction mode, will combine the structure and the crab_ori embedding


    # Choose if you data is a regression or binary classification
    classification = False
    # train = False
    train = True

    # Train your model using the "get_model" function
    if train:
        print(f'Property "{mat_prop}" selected for training')
        model = get_model(data_dir, mat_prop, classification, \
                          transfer=Transfer , verbose=True, save_path=save_model_path,\
                          model_name=save_model_name, embedding_loss=embedding_loss, \
                          pretrained=pretrain, finetune=finetune, embedding_path=embedding_path,\
                          crab_ori_embedding=crab_ori_embedding)

    cutter = '====================================================='
    first = " "*((len(cutter)-len(mat_prop))//2) + " "*int((len(mat_prop)+1)%2)
    last = " "*((len(cutter)-len(mat_prop))//2)
    print('=====================================================')
    print(f'{first}{mat_prop}{last}')
    print('=====================================================')
    print('calculating train mae')
    model_train, mae_train, embeddings0 = save_results(data_dir, mat_prop, classification,
                                          'train.csv', verbose=False, model_name=save_model_name,
                                          save_model_path=save_model_path, pretrained=pretrain,
                                          crab_ori_embedding=crab_ori_embedding)
    print('-----------------------------------------------------')
    print('calculating val mae')
    model_val, mae_valn, embeddings1 = save_results(data_dir, mat_prop, classification,
                                       'val.csv', verbose=False, model_name=save_model_name,
                                          save_model_path=save_model_path, pretrained=pretrain,
                                          crab_ori_embedding=crab_ori_embedding)
    print('-----------------------------------------------------')
    print('calculating test mae')
    model_test, mae_test, embeddings2 = save_results(data_dir, mat_prop, classification,
                                        'test.csv', verbose=False, model_name=save_model_name,
                                          save_model_path=save_model_path, pretrained=pretrain,
                                          crab_ori_embedding=crab_ori_embedding)
    print('=====================================================')
    # embeddings0.update(embeddings1)
    # embeddings0.update(embeddings2)
    # embeddings = embeddings0
    # import sys
    # from pympler import asizeof
    # print(len(embeddings), sys.getsizeof(embeddings))
    # print(asizeof.asizeof(embeddings))
    # import pickle
    # with open(f'/home/zd/zd/teaching_net/CrabNet/predict_structurEmbedding/jarvis_{prop}/embeddings/jarvis_crabnet_oriEmbedding.pkl', 'wb') as f:
    #     pickle.dump(embeddings, f, protocol=4)
    # print('embedding saved')
