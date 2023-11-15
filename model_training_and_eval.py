import pickle 
import matplotlib.pyplot as plt
import numpy as np
import cv2, os, json
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from tensorflow.keras import mixed_precision, metrics, optimizers
from tensorflow.keras.metrics import Precision, Recall, IoU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import UnetModels
print(tf.__version__)

os.environ["CUDA_VISIBLE_DEVICES"]="0" # "5,6,7"
policy = mixed_precision.Policy('float32')
mixed_precision.set_global_policy(policy)

class PredictAndVisualizeCallback(Callback):
    def __init__(self, model, test_generator, model_name='test_model', num_samples=3, interval=1):
        self.model = model
        self.test_generator = test_generator
        self.num_samples = num_samples
        self.model_name = model_name
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.interval:
            return False
        
        if logs is None:
            logs = {}

        
        m=0
        plt.figure(figsize=(12, 12))
        for i in range(self.num_samples):
            try:
                
                # Make predictions for the batch
                # Get the next batch from the validation generator
                image_batch, mask_batch = next(self.test_generator)
                predictions = self.model.predict(image_batch)
                #sample_indices = np.random.choice(len(image_batch), self.num_samples, replace=False)

                # Loop through the batch and plot individual samples
                image = image_batch[0]
                mask = mask_batch[0]
                y_pred = predictions[0]

                # Plot the input image, ground truth mask, and predicted mask
                plt.subplot(3, 3, m*3+1)
                plt.imshow(image[:, :, 0], cmap='gray')
                plt.title('Input Image')

                plt.subplot(3, 3, m*3+2)
                plt.imshow(mask[:, :, 0], cmap='gray')
                plt.title('Ground Truth Mask')

                plt.subplot(3, 3, m*3+3)
                plt.imshow(y_pred[:, :, 0], cmap='gray')
                plt.title('Predicted Mask')

                plt.tight_layout()
                m+=1
            except StopIteration:
                # StopIteration indicates the end of an epoch; reset the generator
                self.validation_generator.on_epoch_end()
                break

        plt.savefig(f"{self.model_name}_pred_e{(epoch+1):03d}.png")
        plt.close()


# Define param
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1
INPUT_SHAPE = (IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS)
PICKLE_PATH = 'haglag_imgs_and_Mmyo_0_15_validation.p'

def data_generators(input_shape = (256,256,1),
                    PICKLE_PATH = 'haglag_imgs_and_Mmyo_0_15_validation.p',
                    batch_size = 10,
                    seed = 1,
                    augmentation=None,
                    ):
    
    # Load data
    with open(PICKLE_PATH, "rb") as input_file:
        data = pickle.load(input_file)
    _ids = data.pop("id")
    for (key, value) in data.items():
        number, height, width = value.shape
        if 'Mmyo' in key:
            value = value*255
        data[key] = value.reshape(number, height, width, input_shape[2])

    # Data generators
    if augmentation is None:
        data_aug_opts = dict(
                rescale=1./255,
                rotation_range=30,
                width_shift_range=0.15, 
                height_shift_range=0.15, 
                # zoom_range=0.2, 
                horizontal_flip=1,
                vertical_flip=1,
                fill_mode='constant', #'nearest'
                cval=0.0,
                )
    else:
        data_aug_opts = augmentation

    datagen_opts = dict(
            rescale=1./255.,
    )

    flow_args = dict(
            seed=seed,                  # Set the seed for reproducibility
            batch_size=batch_size,      # Specify the batch size
        )

    data_augmentor = ImageDataGenerator(**data_aug_opts)
    data_generator = ImageDataGenerator(**datagen_opts)

    train_generator = zip(
        data_augmentor.flow(data['train images'], **flow_args),
        data_augmentor.flow(data['train Mmyo'], **flow_args)
    )

    validation_generator = zip(
        data_generator.flow(data['validation images'], **flow_args),
        data_generator.flow(data['validation Mmyo'], **flow_args)
    )

    test_generator = zip(
        data_generator.flow(data['test images'], **flow_args),
        data_generator.flow(data['test Mmyo'], **flow_args)
    )

    test_set = [
        data['test images']*(1./255.), 
        data['test Mmyo']*(1./255.),
    ]

    set_size = (
        len(data['train images']),
        len(data['validation images']),
        len(data['test images']),
        )

    return train_generator, validation_generator, test_generator, test_set, set_size


def dice_loss(y_true, y_pred):

    def dice_coefficient(y_true, y_pred, smooth=1.0):
        intersection = tf.reduce_sum(tf.math.multiply(y_true,y_pred))
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice

    return 1.0 - dice_coefficient(y_true, y_pred)

# These functions are from A.Kregnes, and have been modified slightly due to the differences in formatting bewteen the projects
def F1(output, target):
    npoutput, nptarget = output.astype(bool), target.astype(bool)
    nptarget = nptarget.flatten()
    npoutput = npoutput.flatten()

    f1 = f1_score(nptarget, npoutput, average=None)
    return f1

def jaccard(output, target):
    npoutput, nptarget = output.astype(bool), target.astype(bool)

    intersection = np.logical_and(npoutput, nptarget)
    union = np.logical_or(npoutput, nptarget)
    jacc = intersection.sum() / float(union.sum())
    return jacc

def dice(output, target):
    npoutput, nptarget = output, target
    npoutput = npoutput.flatten()
    nptarget = nptarget.flatten()

    intersection = np.logical_and(nptarget, npoutput)
    A = np.sum(npoutput)
    B = np.sum(nptarget)

    dice = 2 * (intersection.sum() + 1) / (A + B + 1)
    return dice

def train_model(model, 
                train_generator, 
                validation_generator, 
                settings, 
                verbose=True):
    '''
    The settings dictionary must contain the following keys:
    - 'name': Name of the model.
    - 'control_metric': Metric used to control training behavior, e.g., "val_loss".
    - 'learning_rate': Learning rate for the optimizer.
    - 'max_epochs': Maximum number of epochs for training.
    - 'batch_size': Size of batches for training.
    - 'callbacks': List of callbacks for the model during training.
    - 'metrics': List of metrics for evaluating the model.
    - 'val_length': Total number of validation samples.
    - 'train_length': Total number of training samples.
    - 'training_set': Name or description of the training dataset.
    - 'model': Information or name of the model architecture.
    - 'early_stopping': Details or configuration for early stopping.
    - 'patience': Number of epochs with no improvement to stop training.

    Note: The 'train_length' and 'val_length' are used to calculate 'steps_per_epoch' and 'validation_steps' respectively.
    '''

    name = settings['name']
    control_metric = settings['control_metric']
    learning_rate = settings['learning_rate']
    epochs = settings['max_epochs']
    callbacks = settings['callbacks']
    metrics = settings['metrics']
    validation_steps = settings['val_length']/settings['batch_size']
    steps_per_epoch = settings['train_length']/settings['batch_size']

    try:
        os.mkdir(settings['folder'])
    except OSError as e:
        print(e) if verbose else ''

    model.compile(
        optimizer=tf.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss=dice_loss, #weighted_bce_dice_loss, #wce(33.1),  #dice_loss,
        metrics=metrics
    )

    result = model.fit(
        x = train_generator,
        epochs = epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data = validation_generator,
        validation_steps = validation_steps,
        callbacks = callbacks
    )

    model_info = {
        'description': 'Training results for the UNet models',
        'dataset': settings['data_set'],
        'name': name,
        'model': settings['model'],
        'initial_filters': settings['initial_filters'],
        'dropout_rate': settings['dropout_rate'],
        'control_metric': control_metric,
        'early_stopping': settings['early_stopping'],
        'stopping_patience': settings['patience'],
        'stopping_epoch': len(result.history[control_metric]),
        # 'metrics': metrics,
        'learning_rate': learning_rate,
        'batch_size': settings['batch_size'],
        'max_epochs': epochs,       
    }
    result.history['info'] = model_info
    return result

def plot_history(history: dict, metrics: list=['accuracy', 'loss'], savepath: str='ModelHistoryPlot.png', title: str='Model training metrics pr epoch'):
    if len(metrics) != 2:
        print("Can only generate plots for 2 metrics.")
        return


    epochs = range(1, len(history[metrics[0]])+1)
     # Create a single figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(title)
    

    # Plot accuracy on the first subplot
    ax1.plot(epochs, history[metrics[0]], label=metrics[0], color='b')
    ax1.plot(epochs, history["val_"+metrics[0]], label='val_'+metrics[0], color='g')
    # ax1.set_title(title)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(metrics[0])
    ax1.legend(loc='lower right')
    ax1.grid(True)

    # Plot loss on the second subplot
    ax2.plot(epochs, history[metrics[1]], label=metrics[1], color='b')
    ax2.plot(epochs, history["val_"+metrics[1]], label='val_'+metrics[1], color='g')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(metrics[1])
    if min(history["val_" +metrics[1]]) > 4:
        ymax = 40.0
        ymin = 0.0
    elif max(history["val_" +metrics[1]]) > 2:
        ymax = 4.0
        ymin = 0.0
    else:
        ymax = 1.0
        ymin = 0.0
    ax2.set_ylim(top=ymax, bottom=ymin)
    ax2.legend(loc='upper right')
    ax2.grid(True)

    # Adjust space between subplots
    plt.tight_layout()

    plt.savefig(savepath)
    plt.close(fig)


def pred_and_plot(unet_model, test_set, title='test', save_path=None, random_samples=False):

    # Predict using the generator
    preds_test = unet_model.predict(test_set[0], verbose=0)
    preds_test_t = (preds_test > 0.5).astype(np.uint8)
    
    num_samples = len(preds_test_t)

    if random_samples:
        ix = np.random.choice(num_samples, size=5, replace=False)
    else:
        ix = [6, 86, 114, 96, 98]
    
    fig = plt.figure(figsize=(30,15))
    alpha = 0.5  # Transparency factor
    
    for n, i in enumerate(ix, 1):
        # pred_mask = preds_test[i]

        # Add contours from ground truth
        rgb = cv2.cvtColor((test_set[0][i]*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        cont, _ = cv2.findContours(test_set[1][i].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_cont = cv2.drawContours(rgb.copy(), cont, -1, (0,255,79), -1)
        img_cont = cv2.addWeighted(img_cont, alpha, rgb.copy(), 1 - alpha, 0)
        img_cont = cv2.drawContours(img_cont, cont, -2, (255,0,0), 1)
        
        # Add contours from predicted myocardium
        pred_cont, _ = cv2.findContours(preds_test_t[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_pred_cont = cv2.drawContours(rgb.copy(), pred_cont, -2, (0,255,79), -1)
        img_pred_cont = cv2.addWeighted(img_pred_cont, alpha, rgb.copy(), 1 - alpha, 0)
        img_pred_cont = cv2.drawContours(img_pred_cont, pred_cont, -1, (255,0,0), 1)
        
        plt.subplot(2, 5, n)
        if n == 3:
            plt.title('Original Images with Myocardium Mask', fontsize=40)
        plt.imshow(img_cont)
        
        plt.subplot(2, 5, 5+n)
        if n == 3:
            plt.title(f'Predicted Masks, {title}', fontsize=40)
        plt.imshow(img_pred_cont)
        
        for ax in fig.axes:
            ax.axis('off')
        
    plt.savefig(save_path) if save_path is not None else plt.savefig(os.path.join(os.getcwd(), 'Figures/PredictionPlot.png'))
    plt.close(fig)



def model_evaluation(model, test_generator, model_name, model_weight, verbose=False):
    
    all_preds = []
    all_masks = []

    for batch_images, batch_masks in test_generator:
        preds = model.predict(batch_images, verbose=0)
        all_preds.append((preds > 0.5))
        all_masks.append(batch_masks)
        if (len(all_masks)*len(batch_images))>117:
            break
    
    # Concatenate results from all batches
    all_preds_concat = np.concatenate(all_preds).astype(np.bool_).flatten()
    all_masks_concat = np.concatenate(all_masks).astype(np.bool_).flatten()

    prediction_eval = dict(
            precision = round(precision_score(all_masks_concat, all_preds_concat),3),
            recall = round(recall_score(all_masks_concat, all_preds_concat),3),
            specificity = round(recall_score(all_masks_concat, all_preds_concat, pos_label=0),3),
            dice = round(dice(all_preds_concat, all_masks_concat),3),
            jaccard = round(jaccard(all_preds_concat, all_masks_concat),3),
            f1 = np.round(np.array(F1(all_preds_concat, all_masks_concat)),3),
    )
    

    prediction_result = f"""
---------------------------------------------
####### {model_name} #######
######### {model_weight} #########
---------------------------------------------
Precision:      {prediction_eval['precision']}
Recall:         {prediction_eval['recall']}
Specificity:    {prediction_eval['specificity']}
Dice:           {prediction_eval['dice']}
Jaccard:        {prediction_eval['jaccard']}
F1-score:       {prediction_eval['f1']}
---------------------------------------------
"""
    print(prediction_result) if verbose else ''
    prediction_eval['report'] = prediction_result
    return prediction_eval

# for use when json.dump with non serializable numpy arrays
def default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()  # Convert numpy array to list
    raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')


def pred_patient_set(model_path, test_set, model_name='Test_model', save_path=None):
    '''
    This function is used to test an already trained unet model on sets of images from test patients.
    Model and weights are loaded from the model path provided.
    The test set should be in this format:
    {'ABC1': [
        [img1, img2, img3, img4, ...],
        [mask1, mask2, mask3, mask4, ...],
        ],
     'DEF2': [
        [img1, img2, img3, img4, ...],
        [mask1, mask2, mask3, mask4, ...],
        ],
     'GHI3': [
        ...
     ]
    }
    Where the dictionary keys are patient ids, and contains a list of two lists, 
    the first list with images, and masks in the second list.
    The image series from patients is each saved in a separate file in a designated folder, 
    or by default in "Figures/Testing/".
    '''
    if save_path is None:
        save_path = os.path.join(os.getcwd(), "Figures/Testing/")
    else:
        save_path = os.path.join(os.getcwd(), save_path)
    
    model = load_model(
        model_path,
        custom_objects={
            'dice_loss': dice_loss, 
            'PredictAndVisualizeCallback': PredictAndVisualizeCallback
            }
    )

    for patient_number, patient_id in enumerate(test_set.keys()):
        print(f"Predicting with {model_name} on {patient_id}")
        imgs, masks = test_set[patient_id]
        series_length = len(imgs)
        rows = series_length // 4 + ((series_length % 4) > 0)


        # Predict using the generator
        preds = model.predict(imgs, verbose=0)
        preds = (preds > 0.5).astype(np.uint8)
    
    
        fig = plt.figure(figsize=(30,rows*10))
        alpha = 0.42  # Transparency factor
        patient_metrics = {}
    
        for i in range(series_length):
            # Add contours from ground truth
            rgb = cv2.cvtColor((imgs[i]*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            cont, _ = cv2.findContours(masks[i].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            img_cont = cv2.drawContours(rgb.copy(), cont, -1, (0,255,79), -1)
            img_cont = cv2.addWeighted(img_cont, alpha, rgb.copy(), 1 - alpha, 0)
            img_cont = cv2.drawContours(img_cont, cont, -2, (255,0,0), 1)
            
            # Add contours from predicted myocardium
            pred_cont, _ = cv2.findContours(preds[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            img_pred_cont = cv2.drawContours(rgb.copy(), pred_cont, -2, (0,255,79), -1)
            img_pred_cont = cv2.addWeighted(img_pred_cont, alpha, rgb.copy(), 1 - alpha, 0)
            img_pred_cont = cv2.drawContours(img_pred_cont, pred_cont, -1, (255,0,0), 1)
            
            # Calculate Jaccard and Dice of predicted image
            prediction =    np.concatenate(preds[i].copy()).astype(np.bool_).flatten()
            mask =          np.concatenate(masks[i].copy()).astype(np.bool_).flatten()
            jaccard =       round(jaccard(prediction, mask), 3)
            dice =          round(dice(prediction, mask), 3)
            patient_metrics.setdefault("jaccard", []).append(jaccard)
            patient_metrics.setdefault("dice", []).append(dice)
            
            # Generate subplots and insert images
            plt.subplot(rows, 4, i+1)
            plt.xlabel(f"Image {i} | DSC: {dice} | Jaccard: {jaccard}", fontsize=20)
            plt.imshow(img_pred_cont)
            
        fig.text(0.5, 0.98, f"Prediction series of test patient #{patient_number}", ha='center', va='center', fontsize=40)
        fig.text(0.5, 0.95, f"Model: {model_name}", ha='center', va='center', fontsize=28)
        fig.text(0.5, 0.92, f"Patient mean DSC:  {round(sum(patient_metrics['dice'])/len(patient_metrics['dice']), 3)}  | Patient mean Jaccard: {round(sum(patient_metrics['jaccard'])/len(patient_metrics['jaccard']),3)}", ha='center', va='center', fontsize=28)

        plt.tight_layout(h_pad=1.0, rect=[0, 0.03, 1, 1-0.03*rows])
        plt.savefig(os.path.join(save_path, f"Pred_{patient_id}.png"))
        plt.close(fig)


if __name__ == '__main__':
    train_generator, validation_generator, test_generator, test_set, set_size = data_generators()

    model = UnetModels.unet_standard(INPUT_SHAPE, 16, 0.2)

    model_name = 'UNet_Segmentation_Model_Test'
    model_type = 'Standard'
    model_training_set = 'haglag_imgs_and_Mmyo_0_15_validation.p'
    early_stopping_patience = 25
    model_folder = f"Models/{model_type}/{model_name}/"
    model_batch_size = 10
    learning_rate = 5e-5

    settings_example = {
        'name': model_name,
        'folder': model_folder,
        'control_metric': 'val_iou_pos',
        'learning_rate': learning_rate,
        'max_epochs': 101,
        'batch_size': model_batch_size,
        'callbacks': [
            EarlyStopping(monitor='val_iou_pos', 
                            mode='max', 
                            patience=early_stopping_patience, 
                            restore_best_weights=True,
                            verbose=1,
                            ),
            ModelCheckpoint(model_folder+'best_model_val_loss.h5',
                            monitor='val_loss', 
                            mode='min', 
                            save_best_only=True,
                            verbose=1,
                            ),
            ModelCheckpoint(model_folder + 'best_model_val_iou.h5',
                            monitor='val_iou_pos', 
                            mode='max', 
                            save_best_only=True,
                            verbose=1,
                            ),
            ModelCheckpoint(model_folder + 'best_model_precision.h5',
                            monitor='val_precision', 
                            mode='max', 
                            save_best_only=True,
                            verbose=1,
                            ),
            ModelCheckpoint(model_folder + 'best_model_recall.h5',
                            monitor='val_recall', 
                            mode='max', 
                            save_best_only=True,
                            verbose=1,
                            ),
            ModelCheckpoint(model_folder + 'best_model_accuracy.h5',
                            monitor='val_accuracy', 
                            mode='max', 
                            save_best_only=True,
                            verbose=1,
                            ),
            PredictAndVisualizeCallback(model, 
                            test_generator, 
                            model_folder + 'test', 
                            interval=10)
        ],
        'metrics': ['accuracy',
                    Precision(name='precision'),
                    Recall(name='recall'),
                    IoU(num_classes=2, name='iou_neg', target_class_ids=[0]),
                    IoU(num_classes=2, name='iou_pos', target_class_ids=[1])
                    ],
        'val_length': 74,  
        'train_length': 273, 
        'test_lengtg': 117,
        'training_set': model_training_set,
        'model': model_type,
        'early_stopping': f'Enabled with patience {early_stopping_patience}',
        'patience': early_stopping_patience,
        }

    

    result = train_model(model=model,
                         train_generator=train_generator,
                         validation_generator=validation_generator,
                         settings=settings_example)
    with open(model_folder+"history.json", 'w') as f:
        json.dump(result.history, f, indent=4)
    
    # print(result.history['info'])

    best_models = [file for file in os.listdir(model_folder) if file.endswith('.h5')]
    print(best_models)
    eval_summary = ''
    eval_json = dict()
    for m in best_models:
        print('Evaluating', m)
        model = load_model(model_folder+m, custom_objects={'dice_loss': dice_loss, 'PredictAndVisualizeCallback': PredictAndVisualizeCallback})
        
        pred_and_plot(model, test_set, 'test', model_folder+m[:-3]+'_PredPlot.png')
        
        model_eval = model_evaluation(model, test_generator, model_name, m)
        eval_summary+= model_eval['report'] + '\n\n'
        eval_json[m] = model_eval


    
    with open(model_folder+'EvaluationMetrics.txt', 'w') as f:
        f.write(eval_summary)

    # print(eval_json)
    with open(model_folder+'evaluation.json', 'w') as j:
        json.dump(eval_json, j, default=default, indent=4)



