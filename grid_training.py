import UnetModels
import model_training_and_eval as mte
import os, sys, json, pickle, cv2, csv, glob, re, os, math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import mixed_precision
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback

os.environ["CUDA_VISIBLE_DEVICES"]="0" # "5,6,7"
policy = mixed_precision.Policy('float32')
mixed_precision.set_global_policy(policy)


################### GRID PARAMETERS ###################
# Define param
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1
INPUT_SHAPE = (IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS)


DATA_SET = [
    'haglag_imgs_and_Mmyo_0_15_validation.p',
    'haglag_imgs_and_Mmyo_0_075_validation.p',
    'patient_imgs_and_Mmyo_vxvy.p',
]

UNET_MODEL = [
    'MultiRes',
    'Standard',
    'Residual',
]
INITIAL_FILTER = [
    16,
    32,
    64,
]
BATCH_SIZE = [
    4,
    8,
    12,
    16,
    32,
]
LEARNING_RATE = [
    5e-2,
    1e-2,
    5e-3,
    1e-3,
    5e-4,
    1e-4,
    5e-5,
    1e-5,
    5e-6,
    1e-6,
]
DROPOUT = [
    0.0,
    0.1,
    0.3,
    0.5,
]
EARLY_STOPPING = [
    25,
]
LOSS_FUNCTIONS = [
    'dice'
]




if __name__ == '__main__':
    model_number = 0
    seed = 1
    data_set = DATA_SET[0]
    max_epochs = 1000
    control_metric = 'val_loss'

    resume_grid_from = 1 # 1 if start from scratch

    # Initate grid evaluation
    for model_type in UNET_MODEL:
        grid_results = dict()
        grid_list = dict()
        grid_folder = f"Models/TopModelRetraining/{model_type}"
        for batch_size in BATCH_SIZE:
            for initial_filters in INITIAL_FILTER:
                for dropout in DROPOUT:
                    for learning_rate in LEARNING_RATE:

                        model_number += 1
                        print(f"Grid training model number {model_number}:")

                        # If resume training from grid model number:
                        if model_number < resume_grid_from:
                            with open(f"{grid_folder}/GridEvaluation.json", 'r') as j:
                                grid_results = json.load(j)

                            with open(f"{grid_folder}/GridModelInfo.json", 'r') as j:
                                grid_list = json.load(j)

                            continue


                        ##################################################
                        ############# Model design phase ###############
                        ##################################################
                        if model_type == 'Standard':
                            history_plot_metrics = ['iou_pos', 'loss']
                            model = UnetModels.unet_standard(INPUT_SHAPE, initial_filters=initial_filters, dropout_rate=dropout)
                        elif model_type == 'Residual':
                            history_plot_metrics = ['iou_pos', 'loss']
                            model = UnetModels.unet_with_residuals(INPUT_SHAPE, initial_filters=initial_filters, dropout_rate=dropout)
                        elif model_type == 'MultiRes':
                            history_plot_metrics = ['accuracy', 'loss']
                            model = UnetModels.MultiResUnet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
                        else:
                            model = UnetModels.unet_standard(INPUT_SHAPE)
                        

                        ##################################################
                        ############# Data generation phase ##############
                        ##################################################
                        train_generator, validation_generator, test_generator, test_set, set_size = mte.data_generators(
                            input_shape=INPUT_SHAPE,
                            PICKLE_PATH=data_set,
                            batch_size=batch_size,
                            seed=1,
                            augmentation=None, # Default augmentation
                            )

                        ##################################################
                        ############### Model settings ###################
                        ##################################################
                        model_name = f"UNet_{model_type}_MyoSeg_Model_{model_number}"
                        model_folder = f"{grid_folder}/{model_name}/"
                        early_stopping_patience = EARLY_STOPPING[0]                


                        model_settings = {
                            'name': model_name,
                            'folder': model_folder,
                            'control_metric': control_metric,
                            'initial_filters': initial_filters,
                            'dropout_rate': dropout,
                            'learning_rate': learning_rate,
                            'max_epochs': max_epochs,
                            'batch_size': batch_size,
                            'callbacks': [
                                mte.EarlyStopping(monitor=control_metric, 
                                                mode='min', 
                                                patience=early_stopping_patience, 
                                                restore_best_weights=True,
                                                verbose=1,
                                                ),
                                mte.ModelCheckpoint(model_folder+'best_model_val_loss.h5',
                                                monitor='val_loss', 
                                                mode='min', 
                                                save_best_only=True,
                                                verbose=1,
                                                ),
                                mte.ModelCheckpoint(model_folder + 'best_model_val_iou.h5',
                                                monitor='val_iou_pos', 
                                                mode='max', 
                                                save_best_only=True,
                                                verbose=1,
                                                ),
                                mte.ModelCheckpoint(model_folder + 'best_model_accuracy.h5',
                                                monitor='val_accuracy', 
                                                mode='max', 
                                                save_best_only=True,
                                                verbose=1,
                                                ),
                                mte.PredictAndVisualizeCallback(model, 
                                                test_generator, 
                                                model_folder + 'test', 
                                                interval=25)
                            ],
                            'metrics': ['accuracy',
                                        mte.Precision(name='precision'),
                                        mte.Recall(name='recall'),
                                        mte.IoU(num_classes=2, name='iou_neg', target_class_ids=[0]),
                                        mte.IoU(num_classes=2, name='iou_pos', target_class_ids=[1])
                                        ],
                            'val_length': set_size[1],  
                            'train_length': set_size[0], 
                            'test_lengtg': set_size[2],
                            'data_set': data_set,
                            'model': model_type,
                            'early_stopping': f'Enabled with {control_metric} with patience of {early_stopping_patience}',
                            'patience': early_stopping_patience,
                            }


                        print(f"""
Current Model:
{model_settings['name']}
{model_settings['model']}
{model_settings['early_stopping']}
Learning rate: {model_settings['learning_rate']}
Batch size: {model_settings['batch_size']}
Initial filters: {model_settings['initial_filters']}
Dropout rate: {model_settings['dropout_rate']}
""")

                        #################################################
                        ############ Model training phase ###############
                        #################################################
                        result = mte.train_model(
                            model=model,
                            train_generator=train_generator,
                            validation_generator=validation_generator,
                            settings=model_settings
                            )

                        history = result.history
                        last_epoch = history['info']['stopping_epoch']
                        model_info = dict(
                            model_name = model_name,
                            model_folder = model_folder,
                            data_set = data_set,
                            model_type = model_type,
                            batch_size = batch_size,
                            learning_rate = learning_rate,
                            dropout = dropout,
                            initial_filters = initial_filters,
                            max_epochs = max_epochs,
                            control_metric = control_metric,
                            patience = early_stopping_patience,
                            stopping_epoch = last_epoch
                        )
                        grid_list[str(model_number)] = model_info
                        
                        ##################################################
                        ############ Model evaluation phase ##############
                        ##################################################
                        # with open(model_folder+"history.json", 'r') as f:
                        #     history = json.load(f)

                        mte.plot_history(
                            history=history, 
                            metrics=history_plot_metrics, 
                            savepath=model_folder + f'HistoryCurve_{model_type}_{model_number}.png', 
                            title=f"UNet {model_type} Model no {model_number}")


                        best_models = [file for file in os.listdir(model_folder) if file.endswith('.h5')]
                        eval_summary = ''
                        eval_json = dict()
                        for m in best_models:
                            print('Evaluating', m)
                            model = mte.load_model(
                                model_folder+m, 
                                custom_objects={
                                    'dice_loss': mte.dice_loss, 
                                    'PredictAndVisualizeCallback': mte.PredictAndVisualizeCallback
                                    }
                                )
                            
                            mte.pred_and_plot(
                                model, 
                                test_set, 
                                f"Model number {model_number} with {m[11:-3]}", 
                                f"{model_folder}{model_type}{model_number}_{m[11:-3]}+_pred.png"
                                )
                            
                            model_eval = mte.model_evaluation(model, test_generator, model_name, m)
                            eval_summary+= model_eval['report'] + '\n\n'
                            eval_json[m] = model_eval
                        
                            # Combined evaluation of grid test
                            grid_results[str(model_number)+m[11:-3]] = {
                                'precision':    model_eval['precision'],
                                'recall':       model_eval['recall'],
                                'specificity':  model_eval['specificity'],
                                'dice':         model_eval['dice'],
                                'jaccard':      model_eval['jaccard'],
                                'f1-score':     model_eval['f1'],
                                }

                        ##################################################
                        ############ Write results to files ##############
                        ##################################################
                        
                        with open(model_folder+"history.json", 'w') as f:
                            json.dump(result.history, f, indent=4)

                        with open(model_folder+"model_info.json", 'w') as f:
                           json.dump(model_info, f, indent=4)

                        with open(model_folder+'EvaluationMetrics.txt', 'w') as f:
                            f.write(eval_summary)

                        # print(eval_json)
                        with open(model_folder+'evaluation.json', 'w') as j:
                            json.dump(eval_json, j, default=mte.default, indent=4)

                        with open(f"{grid_folder}/GridEvaluation.json", 'w') as j:
                            json.dump(grid_results, j, default=mte.default, indent=4)

                        with open(f"{grid_folder}/GridModelInfo.json", 'w') as j:
                            json.dump(grid_list, j, indent=4)


        ##################################################
        ############## Show 10 best models ###############
        ##################################################
        # with open(f"{grid_folder}/GridEvaluation.json", 'r') as j:
        #     grid_results = json.load(j)

        # with open(f"{grid_folder}/GridModelInfo.json", 'r') as j:
        #     grid_list = json.load(j)


        # Get the top models based on the jaccard measurement
        number_of_models = 10
        sorted_models = sorted(grid_results.items(), key=lambda item: item[1]['jaccard'], reverse=True)
        top_jaccard_models = [model[0] for model in sorted_models[:number_of_models]]
        top = list()
        top_metrics = list()
        for m in top_jaccard_models:
            match = re.match(r'^\d+', m)
            mno = str(match.group())
            model_string = [
                m,
                grid_list[mno]['model_type'],
                str(grid_list[mno]['initial_filters']),
                str(grid_list[mno]['batch_size']),
                str(grid_list[mno]['learning_rate']),
                str(grid_list[mno]['dropout']),
                str(grid_list[mno]['stopping_epoch']),
                str(grid_results[m]['jaccard']),
                ]
            top.append(model_string)
            metric_string = [
                m,
                grid_list[mno]['model_type'],
                str(grid_results[m]['precision']),
                str(grid_results[m]['recall']),
                str(grid_results[m]['specificity']),
                str(grid_results[m]['dice']),
                str(grid_results[m]['jaccard']),
                str(grid_results[m]['f1-score'][0]),
                str(grid_results[m]['f1-score'][1]),
            ]
            top_metrics.append(metric_string)

        # Calculate the maximum width of each column
        max_widths = [max(len(entry[i])+7 for entry in top) for i in range(len(top[0]))]
        header = f"{'No':<4} {'Name':<{max_widths[0]}} {'ModelType':<{max_widths[1]}} {'Filters':<{max_widths[2]}} {'Batch':<{max_widths[3]}} {'LearnRate':<{max_widths[4]}} {'Dropout':<{max_widths[5]}} {'Epochs':<{max_widths[6]}} {'Jaccard':<{max_widths[7]}}"
        separator = ' '.join('-' * width for width in max_widths)

        # Add each row to the table string
        table_str = f"{header}\n---- {separator}\n"
        for i, row in enumerate(top, 1):
            table_str += f"{i:<4} {row[0]:<{max_widths[0]}} {row[1]:<{max_widths[1]}} {row[2]:<{max_widths[2]}} {row[3]:<{max_widths[3]}} {row[4]:<{max_widths[4]}} {row[5]:<{max_widths[5]}} {row[6]:<{max_widths[6]}} {row[7]:<{max_widths[7]}}\n"

        # Print the table
        print(table_str)

        # Save table
        with open(f"{grid_folder}/BestResults.txt", "w") as f:
            f.write(table_str)

        # Calculate the maximum width of each column
        max_widths = [max(len(entry[i])+7 for entry in top_metrics) for i in range(len(top_metrics[0]))]
        header = f"{'No':<4} {'Name':<{max_widths[0]}} {'ModelType':<{max_widths[1]}} {'Precision':<{max_widths[2]}} {'Recall':<{max_widths[3]}} {'Specificity':<{max_widths[4]}} {'Dice':<{max_widths[5]}} {'Jaccard':<{max_widths[6]}} {'F1-score BG':<{max_widths[7]}} {'F1-score FG':<{max_widths[8]}}"
        separator = ' '.join('-' * width for width in max_widths)

        # Add each row to the table string
        table_str = f"{header}\n---- {separator}\n"
        for i, row in enumerate(top_metrics, 1):
            table_str += f"{i:<4} {row[0]:<{max_widths[0]}} {row[1]:<{max_widths[1]}} {row[2]:<{max_widths[2]}} {row[3]:<{max_widths[3]}} {row[4]:<{max_widths[4]}} {row[5]:<{max_widths[5]}} {row[6]:<{max_widths[6]}} {row[7]:<{max_widths[7]}} {row[8]:<{max_widths[8]}}\n"

        # Print the table
        print(table_str)

        # Save table
        with open(f"{grid_folder}/BestResultsMetrics.txt", "w") as f:
            f.write(table_str)

        # Aggregating values for each metric
        grid_metrics = {}
        for key, value in grid_results.items():
            if value['jaccard'] < 0.5:
                continue
            for metric, metric_value in value.items():
                if metric == "f1-score":
                    grid_metrics.setdefault("f1-score bg", []).append(metric_value[0])
                    grid_metrics.setdefault("f1-score fg", []).append(metric_value[1])
                else:
                    grid_metrics.setdefault(metric, []).append(metric_value)

        # Calculating mean and variance for each metric
        grid_statistics = {}
        for metric, values in grid_metrics.items():
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std_dev = math.sqrt(variance)
            grid_statistics[metric] = {"mean": mean, "std_dev": std_dev, "max": max(values), "min": min(values)}

        # Formatting and printing the table
        table_str = f"{'Metric':<25} {'Max':<15} {'Min':<15} {'Mean':<15} {'SD':<15}\n"
        table_str += f"{'-'*25} {'-'*15} {'-'*15} {'-'*15} {'-'*15}\n"

        for metric, values in grid_statistics.items():
            table_str += f"{metric:<25} {values['max']:<15.4f} {values['min']:<15.4f} {values['mean']:<15.4f} {values['std_dev']:<15.4f}\n"

        with open(f"{grid_folder}/GridStatistics_exclude.txt", "w") as f:
            f.write(table_str)