import os
import csv
import time
from tempfile import TemporaryFile

# import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import metrics, callbacks
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras import backend as K

import numpy as np

from matplotlib import pyplot as plt


proj_name = 'Credit Card Fraud Detection'
models_path = 'models'
callbacks_path = os.path.join(models_path, 'callbacks')
hyperas_path = os.path.join(models_path, 'hyperas')
datasets_path = 'extracted datasets'
k_fold_prefix = 'K-fold '


# tf.config.experimental_connect_to_host('grpc://' + os.environ['COLAB_TPU_ADDR'])
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver('grpc://' + os.environ['COLAB_TPU_ADDR'])
# tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.TPUStrategy(resolver)

# tf.compat.v1.disable_eager_execution()


best_f1 = -float('inf')
i = 0
reset_logs = False


compile_args = {
    'optimizer': 'adam',
    'loss': 'binary_crossentropy',
    'metrics': [
#         'accuracy',
#         metrics.TruePositives(),
#         metrics.FalseNegatives(),
        metrics.Recall(),
        metrics.Precision(),
        tfa.metrics.F1Score(num_classes=2, average='micro', threshold=0.5),
        metrics.AUC(curve='ROC', name='AUC_ROC'),
        metrics.AUC(curve='PR', name='AUC_PR')
    ]
}


def gen_callback(model_name=None, monitor='val_f1_score', mode='max', 
                 patience=[10, 5], verbose=1, idx=(0,)):
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            restore_best_weights=True,
            patience=patience[0],
            verbose=verbose,
        ),
#         callbacks.ModelCheckpoint(
#             filepath=os.path.join(callbacks_path, f'{model_name}.h5'),
#             monitor=monitor,
#             mode=mode,
#             save_best_only=True,
#             verbose=verbose,
#         ),
        callbacks.ReduceLROnPlateau(
            monitor=monitor,
            patience=patience[1],
            factor=0.6
        )
    ]
    
    return [callbacks_list[i] for i in idx]


dataset_train_min = None
dataset_train_max = None
dataset_train_mean = None
dataset_train_std = None

def load_data(normalization='z-score'):
    global dataset_min ,dataset_max ,dataset_train_mean ,dataset_train_std
    dataset_path = os.path.join(r'D:\Hosein\Projects\0.Downloads\Datasets\Credit Card Fraud Detection' ,'creditcard.csv')

    dataframe = []
    with open(dataset_path ,'r') as file:
        csv_reader = csv.reader(file ,delimiter=',')
        for row in csv_reader:
            dataframe.append(row)

    dataframe = dataframe[1:]

    data = np.asarray(dataframe ,dtype='float64')
    labels = data[: ,-1].astype('bool')
    data = data[: ,1:-1]
    
    data = data.reshape((data.shape[0], 1 ,data.shape[-1]))

    splitter = int(data.shape[0]*(2/3))
    x_train = data[:splitter]
    y_train = labels[:splitter]
    x_test = data[splitter:]
    y_test = labels[splitter:]
    
    dataset_train_min = np.amin(x_train ,axis=0)
    dataset_train_max = np.amax(x_train ,axis=0)
    dataset_train_mean = x_train.mean(axis=0)
    dataset_train_std = x_train.std(axis=0)

    if normalization == 'z-score':
        x_train -= dataset_train_mean
        x_train /= dataset_train_std
        x_test -= dataset_train_mean
        x_test /= dataset_train_std

    elif normalization == 'min-max':
        x_train = (x_train - dataset_train_min) / (dataset_train_max - dataset_train_min)
        x_test = (x_test - dataset_train_min) / (dataset_train_max - dataset_train_min)
        
    else:
        print('Without normalization')
    
    return x_train ,y_train ,x_test ,y_test


def get_dataset_stats():
    if dataset_min is not None:
        return dataset_min ,dataset_max ,dataset_train_mean ,dataset_train_std
    
    return None


def get_windowed_seq(data ,labels ,w=4):
    zero_class_num = len(labels[labels==0])
    one_class_num = len(labels[labels==1])
    minority_class_num = zero_class_num if zero_class_num < one_class_num else one_class_num
    mijority_class_num = zero_class_num if zero_class_num > one_class_num else one_class_num
    minority_class_value = 0 if zero_class_num < one_class_num else 1
    
    if minority_class_num * w > data.shape[0]:
        print('Given window size is wrong.')
        return None
    
    data = data.copy()
    data = data.reshape((data.shape[0] ,data.shape[-1]))
    
    windowed_seq = np.zeros((minority_class_num ,w ,data.shape[-1]))
    minority_indices = np.asarray(range(labels.shape[0]))[labels == minority_class_value]
    
    for i ,ind in enumerate(minority_indices):
        windowed_seq[i ,: ,:] = data[ind-w+1:ind+1 ,:]
    
    return windowed_seq


def k_fold_validation(model, train_data, compile_args, test_data=None, callbacks=None,
                      epochs=10, batch_size=128, verbose=1, k=3, for_epoch_search=False):
    assert len(train_data[0]) == len(train_data[1])
    if test_data:
        assert len(test_data[0]) == len(test_data[1])
        
    val_size = train_data[0].shape[0] // k
    validations = []
    
    for fold in range(k):
        print(f'\n\nFold : {fold}')
        K.clear_session()

        x_train = np.concatenate(
            [train_data[0][: fold * val_size],
            train_data[0][(fold+1) * val_size: ]]
        )
        y_train = np.concatenate(
            [train_data[1][: fold * val_size],
            train_data[1][(fold+1) * val_size: ]]
        )
        x_val = train_data[0][fold * val_size: (fold+1) * val_size]
        y_val = train_data[1][fold * val_size: (fold+1) * val_size]

        model_k = clone_model(model)
        model_k.compile(**compile_args)
        if not for_epoch_search:
            model_k.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose)

            print(f'\nValidating Fold {fold}')
            val_score = model_k.evaluate(x_val, y_val, verbose=verbose)
            validations.append(val_score)     
        else:
            val_history = model_k.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, 
                                      verbose=verbose, validation_data=(x_val, y_val)).history
            
            val_history = [[value for value in val_history.get(key)] for key in val_history]
            validations.append(val_history)

    if test_data:
        print('\n\nTraining Final model')
        K.clear_session()

        model.compile(**compile_args)
        model.fit(train_data[0], train_data[1], epochs=epochs, batch_size=batch_size, verbose=verbose)

        print('\nEvaluating Final model')
        test_score = model.evaluate(test_data[0], test_data[1], verbose=verbose)

    val_avg = np.average(validations, axis=0)
    model_metrics = model_k.metrics_names
    if for_epoch_search:
        model_metrics += ['val_'+metric for metric in model_metrics]
    
    return (dict(zip(model_metrics, val_avg.tolist())), dict(zip(model_metrics, test_score)) if test_data else None)


def k_fold_prediction(model, train_data, compile_args, callbacks=None,
                      epochs=10, batch_size=128, verbose=1, k=3):
    assert len(train_data[0]) == len(train_data[1])
        
    val_size = train_data[0].shape[0] // k
    preds = []
    
    for fold in range(k):
        print(f'\n\nFold : {fold}')
        K.clear_session()

        x_train = np.concatenate(
            [train_data[0][: fold * val_size],
            train_data[0][(fold+1) * val_size: ]]
        )
        y_train = np.concatenate(
            [train_data[1][: fold * val_size], 
            train_data[1][(fold+1) * val_size: ]]
        )
        x_val = train_data[0][fold * val_size: (fold+1) * val_size]

        model_k = clone_model(model)
        model_k.compile(**compile_args)
        
        model_k.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose)
        preds_k = model_k.predict(x_val, verbose=verbose)
        preds.append(preds_k)     
    
    return np.concatenate(preds)


start=None
def tic():
    global start
    start = time.time()
    return

def toc(title=''):
    global start
    interval = time.time() - start
    print(f'\n{title+" : " if title else ""}{int(interval // 3600)} Hour(s) & {int((interval // 60) % 60)} Minute(s) & {int(interval % 60)} Second(s)')


def plot_history(history, range_=(0, None), max_ind=float('inf')):
    range_ = slice(*range_)
    metrics = list(history.keys())
    has_val = history.get('val_'+metrics[0])

    for itr, metric in enumerate(history.keys()):
        itr += 1
        if (itr > len(history.keys()) / 2 and has_val) or itr > max_ind:
            break

        plt.figure(itr)
        
        epochs = range(1, len(history.get(metric))+1)[range_]
        plt.plot(epochs ,history.get(metric)[range_], label='Training')
        if has_val:
            plt.plot(epochs, history.get('val_'+metric)[range_], label='Validation', marker='v')
    
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.show()


def measure_metrics(model ,x_data ,y_true):
    y_pred = model.predict(x_data)

    bAccuracy = metrics.BinaryAccuracy()
    bAccuracy.update_state(y_true ,y_pred)
    print('BinaryAccuracy : ' ,bAccuracy.result().numpy())
    
    recall = metrics.Recall()
    recall.update_state(y_true ,y_pred)
    print('Recall : ' ,recall.result().numpy())
    
    precision = metrics.Precision()
    precision.update_state(y_true ,y_pred)
    print('Precision : ' ,precision.result().numpy())
    
    recall = recall.result().numpy()
    precision = precision.result().numpy()
    f1 = (2*precision*recall) / (precision+recall+K.epsilon())
    print('F1 : ' ,f1)
    
    tp = metrics.TruePositives()
    tp.update_state(y_true ,y_pred)
    print('TruePositives : ' ,tp.result().numpy())
    
    fn = metrics.FalseNegatives()
    fn.update_state(y_true ,y_pred)
    print('FalseNegatives : ' ,fn.result().numpy())
    
    tn = metrics.TrueNegatives()
    tn.update_state(y_true ,y_pred)
    print('TrueNegatives : ' ,tn.result().numpy())
    
    fp = metrics.FalsePositives()
    fp.update_state(y_true ,y_pred)
    print('FalsePositives : ' ,fp.result().numpy())
    
    roc = metrics.AUC(curve='ROC')
    roc.update_state(y_true ,y_pred)
    print('AUC-ROC : ' ,roc.result().numpy())
    
    pr = metrics.AUC(curve='PR')
    pr.update_state(y_true ,y_pred)
    print('AUC-PR : ' ,pr.result().numpy())


def measure_metrics2(y_true, y_pred):
    bAccuracy = metrics.BinaryAccuracy()
    bAccuracy.update_state(y_true, y_pred)
    print('BinaryAccuracy : ', bAccuracy.result().numpy())
    
    recall = metrics.Recall()
    recall.update_state(y_true, y_pred)
    print('Recall : ', recall.result().numpy())
    
    precision = metrics.Precision()
    precision.update_state(y_true, y_pred)
    print('Precision : ', precision.result().numpy())
    
    recall = recall.result().numpy()
    precision = precision.result().numpy()
    f1 = (2*precision*recall) / (precision+recall+K.epsilon())
    print('F1 : ', f1)
    
    tp = metrics.TruePositives()
    tp.update_state(y_true, y_pred)
    print('TruePositives : ', tp.result().numpy())
    
    fn = metrics.FalseNegatives()
    fn.update_state(y_true, y_pred)
    print('FalseNegatives : ', fn.result().numpy())
    
    tn = metrics.TrueNegatives()
    tn.update_state(y_true, y_pred)
    print('TrueNegatives : ', tn.result().numpy())
    
    fp = metrics.FalsePositives()
    fp.update_state(y_true, y_pred)
    print('FalsePositives : ', fp.result().numpy())
    
    roc = metrics.AUC(curve='ROC')
    roc.update_state(y_true, y_pred)
    print('AUC-ROC : ', roc.result().numpy())
    
    pr = metrics.AUC(curve='PR')
    pr.update_state(y_true, y_pred)
    print('AUC-PR : ', pr.result().numpy())


def generate_from_vae(decoder, n, latent_dim, verbose=0):
    z_sample = np.random.normal(size=(n, latent_dim), loc=0., scale=1.)
    x_decoded = decoder.predict(z_sample, verbose=verbose)

    return x_decoded #new_samples


def save_samples(arr ,path ,type_):
    if type_ == '.csv':
        np.savetxt(path+type_ ,arr ,delimiter=',')

    elif type_ == '.npy':
        outfile = TemporaryFile()

        with open(path+type_ ,'wb') as file:
            np.save(file ,arr)
            
    else:
        print('Wrong type!')
            
            
def load_samples(path ,type_):
    if type_ == '.csv':
        pass
        
    elif type_ == '.npy':
        with open(path+type_ ,'rb') as file:
            arr = np.load(file ,allow_pickle=True)
            
    else:
        return None
    
    return arr


def test_new_samples(model_path ,samples_path ,cmp_args=compile_args ,epochs=10 ,batch_size=128 ,needed_range=(0 ,None)):
    ind_min ,ind_max = needed_range

    x_train ,y_train ,x_test ,y_test = load_data('')
    dataset_min ,dataset_max ,dataset_train_mean ,dataset_train_std = get_dataset_stats()
#     x_train = x_train.reshape(x_train.shape[0] ,x_train.shape[-1])
#     x_test = x_test.reshape(x_test.shape[0] ,x_test.shape[-1])

#     sampled_frauds = load_samples(samples_path ,'.npy')[ind_min : ind_max ,-1 ,:]
    sampled_frauds = load_samples(samples_path ,'.npy')[ind_min : ind_max]
#     sampled_frauds = sampled_frauds.reshape(sampled_frauds.shape[0] ,sampled_frauds.shape[-1])
#     sampled_frauds = sampled_frauds.reshape(sampled_frauds.shape[0] ,1 ,sampled_frauds.shape[-1])
    sampled_lables = np.ones((sampled_frauds.shape[0] ,) ,dtype='bool')

    x_train = np.concatenate([
        x_train ,
        sampled_frauds
    ])
    y_train = np.concatenate([
        y_train ,
        sampled_lables
    ])

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train -= mean
    x_train /= std
    x_test -= mean
    x_test /= std

    shuffler = np.random.permutation(x_train.shape[0])
    x_train = x_train[shuffler]
    y_train = y_train[shuffler]
    
    model = load_model(model_path)
    print('\n---Testing old model metrics on new trainset :')
    test_metrics(model ,x_train ,y_train)
    print('\n---Evaluating old model :')
    model.evaluate(x_test ,y_test)

    model = clone_model(model)
    model.compile(**cmp_args)
    
    tic()
    print('\n---Training new model :')
    history = model.fit(
        x_train ,
        y_train ,
        epochs=epochs ,
        batch_size=batch_size ,
    #     validation_split=0.2
    ).history
    toc('Training time :')
    plot_history(history)
    
    print('\n---Testing new model metrics on new trainset :')
    test_metrics(model ,x_train ,y_train)
    
    print('\n---Evaluating new model :')
    model.evaluate(x_test ,y_test)
    
    print('\n---Testing new model metrics on testset :')
    test_metrics(model ,x_test ,y_test)
    
    
def save_logs(model_name, i, val_f1, best_f1, search_space, names, where_to='file'):
    txt = f'----Opt Epoch {i} : val_f1={val_f1}, best_f1={best_f1}\n'
    for ss, name in zip(search_space, names):
        txt += f'{name} : {ss}\n'
    txt += '\n'
        
    if where_to == 'file' or where_to == 'both':
        with open(f'models/hyperas/logs/{model_name}.txt', 'at') as f: 
            f.write(txt)
    if where_to == 'print' or where_to == 'both':
        print(txt)

 
