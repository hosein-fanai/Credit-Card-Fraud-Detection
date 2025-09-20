import os

import time


proj_name = "Credit Card Fraud Detection"
datasets_path = "./datasets"
models_path = "./models"
callbacks_path = os.path.join(models_path, "callbacks")
hyperas_path = os.path.join(models_path, "hyperas")
extracted_data_path = "./extracted data"
k_fold_prefix = "K-fold "


best_f1 = -float("inf")
i = 1
reset_logs = False


def create_compile_args(optimizer="adam", loss="binary_crossentropy", metrics_idx=list(range(8))):
    from tensorflow.keras import metrics

    import tensorflow_addons as tfa


    all_metrics = [
        metrics.BinaryAccuracy(),
        metrics.TruePositives(),
        metrics.FalseNegatives(),
        metrics.Recall(),
        metrics.Precision(),
        tfa.metrics.F1Score(num_classes=2, average="micro", threshold=0.5), 
        metrics.AUC(curve="ROC", name="AUC_ROC"),
        metrics.AUC(curve="PR", name="AUC_PR"),
    ]

    compile_args = {
        "optimizer": optimizer,
        "loss": loss,
        "metrics": [all_metrics[metric_idx] for metric_idx in metrics_idx]
    }

    return compile_args


def create_callbacks_list(model_name="no name", monitor="val_f1_score", mode="max", 
                        patience=5, min_delta=1e-3, reducelr_factor=0.6, verbose=1, idx=(0,)):
    from tensorflow.keras import callbacks


    callbacks_list = [
        callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            restore_best_weights=True,
            patience=patience,
            min_delta=min_delta,
            verbose=verbose,
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(callbacks_path, f"{model_name}.h5"),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            verbose=verbose,
        ),
        callbacks.ReduceLROnPlateau(
            monitor=monitor,
            patience=patience,
            min_delta=min_delta,
            factor=reducelr_factor,
            verbose=verbose,
        )
    ]
    
    return [callbacks_list[i] for i in idx]


dataset_train_min = None
dataset_train_max = None
dataset_train_mean = None
dataset_train_std = None

def load_data_ccfd(normalization="z-score"):
    import numpy as np

    import csv


    global dataset_train_min, dataset_train_max, dataset_train_mean, dataset_train_std
    dataset_path = os.path.join(datasets_path, "creditcard.csv")

    dataframe = []
    with open(dataset_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            dataframe.append(row)

    dataframe = dataframe[1:]

    data = np.asarray(dataframe, dtype="float64")
    labels = data[:, -1].astype("bool")
    data = data[:, 1:-1]

    splitter = int(data.shape[0]*(2/3))
    x_train = data[:splitter]
    y_train = labels[:splitter]
    x_test = data[splitter:]
    y_test = labels[splitter:]
    
    dataset_train_min = np.amin(x_train, axis=0)
    dataset_train_max = np.amax(x_train, axis=0)
    dataset_train_mean = x_train.mean(axis=0)
    dataset_train_std = x_train.std(axis=0)

    if normalization == "z-score":
        x_train -= dataset_train_mean
        x_train /= dataset_train_std
        x_test -= dataset_train_mean
        x_test /= dataset_train_std

    elif normalization == "min-max":
        x_train = (x_train - dataset_train_min) / (dataset_train_max - dataset_train_min)
        x_test = (x_test - dataset_train_min) / (dataset_train_max - dataset_train_min)
        
    else:
        print("Without normalization")
    
    return x_train, y_train, x_test, y_test


def load_data_ccfd2023(normalization="z-score"):
    from sklearn.model_selection import train_test_split

    import numpy as np

    import pandas as pd


    global dataset_min, dataset_max, dataset_train_mean, dataset_train_std


    dataset_path = os.path.join(datasets_path, "creditcard_2023.csv")
    dataframe = pd.read_csv(dataset_path)

    labels = dataframe["Class"].to_numpy().astype("bool")
    data = dataframe.drop(["id", "Class"], axis=1).to_numpy().astype("float64")

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, 
                                            stratify=labels, shuffle=True, random_state=42)

    dataset_train_min = np.amin(x_train, axis=0)
    dataset_train_max = np.amax(x_train, axis=0)
    dataset_train_mean = x_train.mean(axis=0)
    dataset_train_std = x_train.std(axis=0)

    if normalization == "z-score":
        x_train -= dataset_train_mean
        x_train /= dataset_train_std
        x_test -= dataset_train_mean
        x_test /= dataset_train_std

    elif normalization == "min-max":
        x_train = (x_train - dataset_train_min) / (dataset_train_max - dataset_train_min)
        x_test = (x_test - dataset_train_min) / (dataset_train_max - dataset_train_min)
        
    else:
        print("Without normalization")
    
    return x_train, y_train, x_test, y_test


def load_data_gcd(normalization="min-max"):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    import numpy as np

    import csv


    dataset_path = os.path.join(datasets_path, "German", "german.data-numeric")

    dataframe = []
    with open(dataset_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            dataframe.append(row[0].replace("   ", ',').replace("  ", ',').replace(' ', ',')[1:-1].split(','))

    data = np.asarray(dataframe, dtype="int")
    labels = data[:, -1] - 1
    dataset = data[:, :-1]

    idx = np.arange(dataset.shape[0])
    np.random.seed(42)
    np.random.shuffle(idx)
    dataset = dataset[idx]
    labels = labels[idx]

    x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.33, 
                                            stratify=labels, shuffle=True, random_state=42)

    if normalization == "min-max":
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    else:
        print("Without normalization")

    return x_train, y_train, x_test, y_test


def get_dataset_stats():
    if dataset_min is not None:
        return dataset_min ,dataset_max ,dataset_train_mean ,dataset_train_std
    
    return None


def k_fold_validation(model, train_data, compile_args, test_data=None, 
                    callbacks=None, epochs=10, batch_size=128, 
                    validation_split=0., k=3, for_epoch_search=False, 
                    verbose=1):
    from tensorflow.keras import optimizers
    from tensorflow.keras.models import clone_model
    from tensorflow.keras import backend as K

    import numpy as np


    assert len(train_data[0]) == len(train_data[1])
    if test_data:
        assert len(test_data[0]) == len(test_data[1])

    if "optimizer" in compile_args and type(compile_args["optimizer"]) != str:
        optimizer_configs = optimizers.serialize(compile_args["optimizer"])

    val_size = train_data[0].shape[0] // k
    validations = []
    
    for fold in range(k):
        if verbose:
            print(f"\n\nFold: {fold}")
        
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
        compile_args_k = compile_args.copy()
        if "optimizer" in compile_args_k and type(compile_args_k["optimizer"]) != str:
            compile_args_k["optimizer"] = optimizers.deserialize(optimizer_configs)

        model_k.compile(**compile_args_k)

        if not for_epoch_search:
            model_k.fit(
                x_train, 
                y_train, 
                epochs=epochs, 
                batch_size=batch_size, 
                callbacks=callbacks, 
                validation_split=validation_split,
                verbose=verbose,
            )

            if verbose:
                print(f"\nValidating Fold {fold}")

            val_score = model_k.evaluate(x_val, y_val, verbose=verbose)
            validations.append(val_score)     
        else:
            val_history = model_k.fit(
                x_train, 
                y_train, 
                epochs=epochs, 
                batch_size=batch_size, 
                callbacks=callbacks, 
                verbose=verbose, 
                validation_data=(x_val, y_val),
            ).history
            
            val_history = [[value for value in val_history.get(key)] for key in val_history]
            validations.append(val_history)

    if test_data:
        if verbose:
            print("\n\nTraining Final model")
        
        K.clear_session()

        model.compile(**compile_args)
        model.fit(
            train_data[0], 
            train_data[1], 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )

        if verbose:
            print("\nEvaluating Final model")

        test_score = model.evaluate(test_data[0], test_data[1], verbose=verbose)

    val_avg = np.average(validations, axis=0)
    model_metrics = model_k.metrics_names
    if for_epoch_search:
        model_metrics += ["val_"+metric for metric in model_metrics]
    
    return (dict(zip(model_metrics, val_avg.tolist())), dict(zip(model_metrics, test_score)) if test_data else None)


def k_fold_prediction(model, train_data, compile_args, callbacks=None,
                      epochs=10, batch_size=128, validation_split=0., 
                      k=3, verbose=1):
    from tensorflow.keras import optimizers
    from tensorflow.keras.models import clone_model
    from tensorflow.keras import backend as K

    import numpy as np


    assert len(train_data[0]) == len(train_data[1])

    if "optimizer" in compile_args and type(compile_args["optimizer"]) != str:
        optimizer_configs = optimizers.serialize(compile_args["optimizer"])
        
    val_size = train_data[0].shape[0] // k
    preds = []
    
    for fold in range(k):
        if verbose:
            print(f"\n\nFold: {fold}")

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
        compile_args_k = compile_args.copy()
        if "optimizer" in compile_args_k and type(compile_args_k["optimizer"]) != str:
            compile_args_k["optimizer"] = optimizers.deserialize(optimizer_configs)

        model_k.compile(**compile_args_k)
        
        model_k.fit(
            x_train, 
            y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split,
            callbacks=callbacks, 
            verbose=verbose
        )

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
    print(f'\n{title+": " if title else ""}{int(interval // 3600)} Hour(s) & {int((interval // 60) % 60)} Minute(s) & {int(interval % 60)} Second(s) & {(interval - int(interval)) * 1000} Milliseconds')


def plot_history(history, range_=(0, None), max_ind=float("inf")):
    from matplotlib import pyplot as plt


    range_ = slice(*range_)
    metrics = list(history.keys())
    has_val = history.get("val_"+metrics[0])

    for itr, metric in enumerate(history.keys()):
        itr += 1
        if (itr > len(history.keys()) / 2 and has_val) or itr > max_ind:
            break

        plt.figure(itr)
        
        epochs = range(1, len(history.get(metric))+1)[range_]
        plt.plot(epochs ,history.get(metric)[range_], label="Training")
        if has_val:
            plt.plot(epochs, history.get("val_"+metric)[range_], label="Validation", marker='v')
    
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())
        plt.show()


def measure_metrics(y_true, y_pred=None, x_data=None, 
                model=None, idx=list(range(10)), verbose=True):
    assert (y_pred is None) != (model is None), "Either input predictions or a model and its X data."
    assert (x_data is None) == (model is None), "A model with X data are required with together."


    from tensorflow.keras import metrics
    from tensorflow.keras import backend as K


    if model:
        y_pred = model.predict(x_data)

    outputs = {}

    if 0 in idx:
        metric = metrics.BinaryAccuracy()
        metric.update_state(y_true, y_pred)
        acc = metric.result().numpy()
        outputs["accuracy"] = acc

        if verbose:
            print("Accuracy:", acc)

    if 1 in idx:
        metric = metrics.Recall()
        metric.update_state(y_true, y_pred)
        recall = metric.result().numpy()
        outputs["recall"] = recall

        if verbose:
            print("Recall:", recall)
    
    if 2 in idx:
        metric = metrics.Precision()
        metric.update_state(y_true, y_pred)
        precision = metric.result().numpy()
        outputs["precision"] = precision

        if verbose:
            print("Precision:", precision)
    
    if 3 in idx:
        f1 = (2*precision*recall) / (precision+recall+K.epsilon())
        outputs["f1"] = f1

        if verbose:
            print("F1:", f1)
    
    if 4 in idx:
        metric = metrics.TruePositives()
        metric.update_state(y_true, y_pred)
        tp = metric.result().numpy()
        outputs["true_positives"] = tp

        if verbose:
            print("True Positives:", tp)
    
    if 5 in idx:
        metric = metrics.FalseNegatives()
        metric.update_state(y_true, y_pred)
        fn = metric.result().numpy()
        outputs["false_negatives"] = fn

        if verbose:
            print("False Negatives:", fn)
    
    if 6 in idx:
        metric = metrics.TrueNegatives()
        metric.update_state(y_true, y_pred)
        tn = metric.result().numpy()
        outputs["true_negatives"] = tn

        if verbose:
            print("True Negatives:", tn)
    
    if 7 in idx:
        metric = metrics.FalsePositives()
        metric.update_state(y_true, y_pred)
        fp = metric.result().numpy()
        outputs["false_positives"] = fp

        if verbose:
            print("False Positives:", fp)
    
    if 8 in idx:
        metric = metrics.AUC(curve="ROC")
        metric.update_state(y_true, y_pred)
        roc = metric.result().numpy()
        outputs["auc_roc"] = roc

        if verbose:
            print("AUC-ROC:", roc)
    
    if 9 in idx:
        metric = metrics.AUC(curve="PR")
        metric.update_state(y_true, y_pred)
        pr = metric.result().numpy()
        outputs["auc_pr"] = pr

        if verbose:
            print("AUC-PR:", pr)

    return outputs


def generate_from_vae(decoder, n, latent_dim, verbose=0):
    import numpy as np


    z_sample = np.random.normal(size=(n, latent_dim), loc=0., scale=1.)
    x_decoded = decoder.predict(z_sample, verbose=verbose)

    return x_decoded #new_samples


def save_samples(arr, path, type_):
    import numpy as np

    from tempfile import TemporaryFile


    if type_ == ".csv":
        np.savetxt(path+type_ ,arr ,delimiter=',')

    elif type_ == ".npy":
        outfile = TemporaryFile()

        with open(path+type_ ,"wb") as file:
            np.save(file ,arr)
            
    else:
        print("Wrong type!")
            
            
def load_samples(path ,type_):
    import numpy as np


    if type_ == ".csv":
        pass
        
    elif type_ == ".npy":
        with open(path+type_ ,"rb") as file:
            arr = np.load(file, allow_pickle=True)
            
    else:
        return None
    
    return arr
    
    
def save_logs(model_name, i, val_f1, best_f1, 
            search_space, names, where_to="file"):
    txt = f"----Opt Epoch {i}: val_f1={val_f1}, best_f1={best_f1}\n"
    for ss, name in zip(search_space, names):
        txt += f"{name}: {ss}\n"
    txt += '\n'

    if where_to == "file" or where_to == "both":
        with open(f"./models/hyperas/logs/{model_name}.txt", "at") as f: 
            f.write(txt)

    if where_to == "print" or where_to == "both":
        print(txt)


def repeat_fit_and_pred_and_eval(model, fit_args, x_test, y_test, 
                                compile_args=None, n_repeat=5, 
                                different_train_data=False, 
                                different_test_data=False):
    import numpy as np


    if compile_args:
        from tensorflow.keras import models, optimizers


        if "optimizer" in compile_args and type(compile_args["optimizer"]) != str:
            optimizer_configs = optimizers.serialize(compile_args["optimizer"])
    else:
        from sklearn.base import clone

    if different_train_data:
        x_train_list = fit_args.get("x", None)
        x_train_list = fit_args["X"] if x_train_list is None else x_train_list

    if different_test_data:
        x_test_list = x_test

    results_dict = {}
    preds_list = []
    probs_list = []
    for i in range(n_repeat):
        print(f"{i+1}/{n_repeat}")

        if compile_args:
            model = models.clone_model(model)
            compile_args = compile_args.copy()

            if "optimizer" in compile_args and type(compile_args["optimizer"]) != str:
                compile_args["optimizer"] = optimizers.deserialize(optimizer_configs)

            model.compile(**compile_args)
        else:
            model = clone(model)

        if different_train_data:
            if fit_args.get("x", None) is None:
                fit_args["X"] = x_train_list[i]
            else:
                fit_args["x"] = x_train_list[i]

        model.fit(**fit_args)

        if different_test_data:
            x_test = x_test_list[i] 


        if compile_args:
            probs = model.predict(x_test, batch_size=fit_args["batch_size"], verbose=0)
            preds = probs.round()
        else:
            preds = model.predict(x_test)
            probs = model.predict_proba(x_test)

        preds_list.append(preds)
        probs_list.append(probs)

        results = measure_metrics(y_test, preds, 
                                verbose=False)

        for key in results:
            if key in results_dict:
                results_dict[key].append(results[key])
            else:
                results_dict[key] = [results[key]]

    for key in results_dict:
        results_list = results_dict[key]
        mean = np.mean(results_list)
        std = np.std(results_list)

        results_dict[key] = (mean, std)

    preds_list = np.array(preds_list)
    probs_list = np.array(probs_list)

    return results_dict, preds_list, probs_list


def hard_voting(preds):
    import numpy as np


    ensemble_preds = np.zeros(preds.shape[0],)
    for i in range(ensemble_preds.shape[0]):
        ensemble_preds[i] = np.bincount(preds[i]).argmax()

    return ensemble_preds


def soft_voting(probs):
    import numpy as np


    ensemble_probs = np.mean(probs, axis=-1)

    return ensemble_probs.round()


def subsets_util(A, subset, best_subset, index, preds_list, 
                y_true, voting_fn, metric_fn, verbose):
    global best_f1


    if subset:
        ensemble_preds = voting_fn(preds_list[:, subset])
        f1 = metric_fn(y_true, ensemble_preds)

        if verbose:
            print(f"---used models' indices: {subset}")
            print(f"--f1: {f1}\n")

        if f1 > best_f1:
            best_f1 = f1
            best_subset = subset.copy()

    for i in range(index, len(A)): 
        subset.append(A[i])
        best_subset = subsets_util(A, subset, best_subset, i + 1, preds_list, 
                                y_true, voting_fn, metric_fn, verbose) 
        subset.pop(-1)

    return best_subset


def search_best_combination(y_true, preds_list, voting_fn, 
                            metric_fn, verbose=0):
    global best_f1

    A = list(range(preds_list.shape[-1]))
    subset, best_subset = [], []
    index = 0

    best_subset = subsets_util(A, subset, best_subset, index, preds_list, 
                            y_true, voting_fn, metric_fn, verbose)

    final_f1 = best_f1
    best_f1 = -float("inf")

    return final_f1, best_subset


