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
    """
    A callback to visualize predictions of a deep learning model at specified epochs during training.

    This callback is used primarily for visual tasks like image segmentation. It enables the visual comparison of 
    predicted masks against the ground truth after each specified number of epochs, aiding in the evaluation and 
    adjustment of the model during training.

    Parameters:
    model (keras.Model): The model on which predictions are made.
    test_generator (Generator): A test data generator that yields batches of input and true masks.
    model_name (str, optional): A name for the model. Defaults to 'test_model'.
    num_samples (int, optional): Number of samples to visualize per batch. Defaults to 3.
    interval (int, optional): Interval of epochs at which visualization occurs. Defaults to 1 (every epoch).

    Methods:
    on_epoch_end(self, epoch, logs=None): Called at the end of each epoch. Visualizes predictions if the 
    epoch number matches the specified interval. Saves the visualizations to a PNG file.
    """
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
    """
    Creates and returns data generators for training, validation, and testing, along with the test dataset and dataset sizes.

    This function initializes data generators with optional data augmentation for deep learning models. It is specifically 
    tailored for image-based tasks, allowing for the input shape, batch size, and augmentation details to be specified.

    Parameters:
    input_shape (tuple, optional): Shape of the input images. Defaults to (256, 256, 1).
    PICKLE_PATH (str, optional): Path to the pickle file containing the dataset. Defaults to 'haglag_imgs_and_Mmyo_0_15_validation.p'.
    batch_size (int, optional): Number of images per batch. Defaults to 10.
    seed (int, optional): Seed for random number generation to ensure reproducibility. Defaults to 1.
    augmentation (dict, optional): Dictionary specifying data augmentation parameters. If None, default augmentation is applied.

    Returns:
    tuple: A tuple containing the following elements in order:
           - train_generator: A generator for training data.
           - validation_generator: A generator for validation data.
           - test_generator: A generator for test data.
           - test_set: A tuple containing test images and their corresponding masks.
           - set_size: A tuple containing the sizes of the training, validation, and test datasets.

    The function works by loading the dataset from a specified pickle file, applying optional data augmentation, and then 
    creating generators for the training, validation, and test datasets. It is crucial for image segmentation or classification tasks.
    """
    
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
    """
    Computes the Dice loss, a measure for the similarity of two samples, typically used in image segmentation.

    Parameters:
    y_true (Tensor): The ground truth labels.
    y_pred (Tensor): The predicted labels.

    Returns:
    float: The computed Dice loss, a value between 0 and 1, where lower values indicate better model performance.

    This function calculates the Dice loss, which is 1 minus the Dice coefficient - a statistic used to gauge the similarity 
    of two sets of data. The Dice coefficient is calculated based on the intersection over the union of the predicted and true 
    labels, with a smoothing term to avoid division by zero. This loss function is particularly useful for handling 
    imbalanced datasets in segmentation tasks.
    """
    def dice_coefficient(y_true, y_pred, smooth=1.0):
        intersection = tf.reduce_sum(tf.math.multiply(y_true,y_pred))
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice

    return 1.0 - dice_coefficient(y_true, y_pred)


def F1(mask, prediction):
# This function is reworked from C. Cappelen et al. project in 2022.
    # This function calculates the F1-score of a prediction.
    # The function expects both prediction and mask to be numpy array of 1 and 0.
    prediction = prediction.astype(bool).flatten()
    mask = mask.astype(bool).flatten()
    f1 = f1_score(mask, prediction, average=None)
    return f1

def jaccard(mask, prediction):
# This function is reworked from C. Cappelen et al. project in 2022.
    # This function calculates the Jaccard of a prediction.
    # The function expects both prediction and mask to be numpy array of 1 and 0.
    mask = mask.astype(bool).flatten()
    prediction = prediction.astype(bool).flatten()

    intersection = np.logical_and(mask, prediction)
    union = np.logical_or(mask, prediction)
    jacc = intersection.sum() / float(union.sum())
    return jacc

def dice(mask, prediction, smooth=1):
# This function is reworked from C. Cappelen et al. project in 2022.
    # This function calculates the DSC (Dice Similarity Coefficient) of a prediction.
    # The function expects both prediction and mask to be numpy array of 1 and 0.
    mask = mask.astype(bool).flatten()
    prediction = prediction.astype(bool).flatten()
    intersection = np.logical_and(mask, prediction)
    A = np.sum(mask)
    B = np.sum(prediction)

    dice = 2 * (intersection.sum() + smooth) / (A + B + smooth)
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
    validation_steps = settings['val_length']/settings['batch_size']
    steps_per_epoch = settings['train_length']/settings['batch_size']

    try:
        os.mkdir(settings['folder'])
    except OSError as e:
        print(e) if verbose else ''

    model.compile(
        optimizer=tf.optimizers.Adam(
            learning_rate=settings['learning_rate']
        ),
        loss=dice_loss, #weighted_bce_dice_loss, #wce(33.1),  #dice_loss,
        metrics=settings['metrics']
    )

    result = model.fit(
        x = train_generator,
        epochs = settings['max_epochs'],
        steps_per_epoch=steps_per_epoch,
        validation_data = validation_generator,
        validation_steps = validation_steps,
        callbacks = settings['callbacks']
    )

    model_info = {
        'description': 'Training results for the UNet models',
        'dataset': settings['data_set'],
        'name': settings['name'],
        'model': settings['model'],
        'initial_filters': settings['initial_filters'],
        'dropout_rate': settings['dropout_rate'],
        'control_metric': settings['control_metric'],
        'early_stopping': settings['early_stopping'],
        'stopping_patience': settings['patience'],
        'stopping_epoch': len(result.history[settings['control_metric']]),
        'learning_rate': settings['learning_rate'],
        'batch_size': settings['batch_size'],
        'max_epochs': settings['max_epochs'],       
    }
    result.history['info'] = model_info
    return result

def plot_history(history: dict, metrics: list=['accuracy', 'loss'], savepath: str='ModelHistoryPlot.png', title: str='Model training metrics pr epoch'):
    """
    Plots the training history of a machine learning model, specifically focusing on two metrics such as accuracy and loss.

    This function takes a history dictionary, typically returned by the training process of a Keras model, and visualizes the 
    progression of specified metrics over the training epochs. It creates a plot with two subplots, one for each metric.

    Parameters:
    history (dict): The training history returned by the model's fit method. It should contain records of the metrics over epochs.
    metrics (list, optional): A list of two metric names to plot. Defaults to ['accuracy', 'loss'].
    savepath (str, optional): File path where the plot will be saved. Defaults to 'ModelHistoryPlot.png'.
    title (str, optional): The title of the plot. Defaults to 'Model training metrics pr epoch'.

    Returns:
    None: The function does not return any value. It saves the plot to the specified path and closes it after saving.

    The function creates a two-subplot figure, plotting the first specified metric and its validation counterpart on one subplot, 
    and the second metric on the other subplot. It includes legends, labels, and grid lines for clarity. The function also handles 
    the y-axis limits for the loss plot to ensure the plot is readable and informative. This function is essential for visualizing 
    model performance over time and diagnosing potential issues like overfitting or underfitting.
    """
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
# This function is reworked from C. Cappelen et al. project in 2022.
    """
    This function applies a trained U-Net model to a test dataset to generate predictions. It then visualizes these predictions 
    by drawing contours on the original images and predicted masks. The function can either select random samples or use a 
    predefined set of indices for visualization.

    Parameters:
    unet_model (Model): The trained U-Net model used for making predictions.
    test_set (tuple): A tuple containing the test images and their corresponding masks.
    title (str, optional): Title for the plot showing predicted masks. Defaults to 'test'.
    save_path (str, optional): Path where the plot will be saved. If None, saves in the current working directory under 'Figures/PredictionPlot.png'.
    random_samples (bool, optional): If True, selects random samples for visualization; otherwise, uses predefined indices. Defaults to False.

    Returns:
    None: The function does not return any value. It saves the visualizations to the specified path or the default location.

    The function predicts masks using the given U-Net model and test dataset. It then overlays these masks and the ground truth 
    masks on the original images, using contours for clear visualization. The visualizations are useful for assessing the model's 
    performance on specific examples and comparing predicted results with actual data.
    """
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
# This function is reworked from C. Cappelen et al. project in 2022.
    """
    Evaluates a trained model on a test dataset and computes various performance metrics.

    This function applies a trained model to a test generator, collects predictions, and compares them against the true masks 
    to compute several evaluation metrics. It is particularly useful for models in segmentation tasks, where precise 
    evaluation is crucial.

    Parameters:
    model (Model): The trained model to be evaluated.
    test_generator (Generator): A generator that yields batches of test images and their corresponding masks.
    model_name (str): Name of the model, used for display purposes.
    model_weight (str): Identifier for the model's weights, used for display.
    verbose (bool, optional): If True, prints a detailed performance report. Defaults to False.

    Returns:
    dict: A dictionary containing the computed precision, recall, specificity, dice coefficient, jaccard index, 
          F1-score, and a formatted string report of these metrics.

    The function iterates over the provided test generator, collecting predictions and true masks. It stops once a 
    specified number of images have been processed. The function then computes precision, recall, specificity, dice 
    coefficient, jaccard index, and F1-score for the predictions. If verbose is True, it prints a detailed report 
    including the model name and weight identifier. This function is essential for quantitatively assessing the 
    performance of segmentation models, providing a comprehensive view of model accuracy and reliability.
    """
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

def default(o):
# for use with json.dump with non serializable numpy arrays
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
    pass
    # train_generator, validation_generator, test_generator, test_set, set_size = data_generators()

    # model = UnetModels.unet_standard(INPUT_SHAPE, 16, 0.2)

    # model_name = 'UNet_Segmentation_Model_Test'
    # model_type = 'Standard'
    # model_training_set = 'haglag_imgs_and_Mmyo_0_15_validation.p'
    # early_stopping_patience = 25
    # model_folder = f"Models/{model_type}/{model_name}/"
    # model_batch_size = 10
    # learning_rate = 5e-5

    # settings_example = {
    #     'name': model_name,
    #     'folder': model_folder,
    #     'control_metric': 'val_iou_pos',
    #     'learning_rate': learning_rate,
    #     'max_epochs': 101,
    #     'batch_size': model_batch_size,
    #     'callbacks': [
    #         EarlyStopping(monitor='val_iou_pos', 
    #                         mode='max', 
    #                         patience=early_stopping_patience, 
    #                         restore_best_weights=True,
    #                         verbose=1,
    #                         ),
    #         ModelCheckpoint(model_folder+'best_model_val_loss.h5',
    #                         monitor='val_loss', 
    #                         mode='min', 
    #                         save_best_only=True,
    #                         verbose=1,
    #                         ),
    #         ModelCheckpoint(model_folder + 'best_model_val_iou.h5',
    #                         monitor='val_iou_pos', 
    #                         mode='max', 
    #                         save_best_only=True,
    #                         verbose=1,
    #                         ),
    #         ModelCheckpoint(model_folder + 'best_model_precision.h5',
    #                         monitor='val_precision', 
    #                         mode='max', 
    #                         save_best_only=True,
    #                         verbose=1,
    #                         ),
    #         ModelCheckpoint(model_folder + 'best_model_recall.h5',
    #                         monitor='val_recall', 
    #                         mode='max', 
    #                         save_best_only=True,
    #                         verbose=1,
    #                         ),
    #         ModelCheckpoint(model_folder + 'best_model_accuracy.h5',
    #                         monitor='val_accuracy', 
    #                         mode='max', 
    #                         save_best_only=True,
    #                         verbose=1,
    #                         ),
    #         PredictAndVisualizeCallback(model, 
    #                         test_generator, 
    #                         model_folder + 'test', 
    #                         interval=10)
    #     ],
    #     'metrics': ['accuracy',
    #                 Precision(name='precision'),
    #                 Recall(name='recall'),
    #                 IoU(num_classes=2, name='iou_neg', target_class_ids=[0]),
    #                 IoU(num_classes=2, name='iou_pos', target_class_ids=[1])
    #                 ],
    #     'val_length': 74,  
    #     'train_length': 273, 
    #     'test_lengtg': 117,
    #     'training_set': model_training_set,
    #     'model': model_type,
    #     'early_stopping': f'Enabled with patience {early_stopping_patience}',
    #     'patience': early_stopping_patience,
    #     }

    

    # result = train_model(model=model,
    #                      train_generator=train_generator,
    #                      validation_generator=validation_generator,
    #                      settings=settings_example)
    # with open(model_folder+"history.json", 'w') as f:
    #     json.dump(result.history, f, indent=4)
    
    # # print(result.history['info'])

    # best_models = [file for file in os.listdir(model_folder) if file.endswith('.h5')]
    # print(best_models)
    # eval_summary = ''
    # eval_json = dict()
    # for m in best_models:
    #     print('Evaluating', m)
    #     model = load_model(model_folder+m, custom_objects={'dice_loss': dice_loss, 'PredictAndVisualizeCallback': PredictAndVisualizeCallback})
        
    #     pred_and_plot(model, test_set, 'test', model_folder+m[:-3]+'_PredPlot.png')
        
    #     model_eval = model_evaluation(model, test_generator, model_name, m)
    #     eval_summary+= model_eval['report'] + '\n\n'
    #     eval_json[m] = model_eval


    
    # with open(model_folder+'EvaluationMetrics.txt', 'w') as f:
    #     f.write(eval_summary)

    # # print(eval_json)
    # with open(model_folder+'evaluation.json', 'w') as j:
    #     json.dump(eval_json, j, default=default, indent=4)



