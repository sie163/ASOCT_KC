import os
import cv2
import glob
import numpy as np
import keras.optimizers as optim
from Vnet_relu import vnet_fpn_relu_SIMO_2
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from utils import *
from skimage import morphology, measure

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


###------------------------------Feature Loss------------------------------###
def dice_coeff_balanced(y_true, y_pred):
    smooth = 0.00001
    y_true_layer0 = y_true[:,:,:,0]
    y_pred_layer0 = y_pred[:,:,:,0]
    y_true_f_0 = K.flatten(y_true_layer0)
    y_pred_f_0 = K.flatten(y_pred_layer0)
    intersection = K.sum(y_true_f_0 * y_pred_f_0)
    score_0 = (2. * intersection + smooth) / (K.sum(y_true_f_0) + K.sum(y_pred_f_0)+ smooth)
    
    y_true_layer1 = y_true[:,:,:,1]
    y_pred_layer1 = y_pred[:,:,:,1]
    y_true_f_1 = K.flatten(y_true_layer1)
    y_pred_f_1 = K.flatten(y_pred_layer1)
    intersection = K.sum(y_true_f_1 * y_pred_f_1)
    score_1 = (2. * intersection + smooth) / (K.sum(y_true_f_1) + K.sum(y_pred_f_1)+ smooth)
    
    return 0.5 * (2 - score_0 - score_1)


def get_border_mask(pool_size, y_true):
    negative = 1 - y_true
    positive = y_true
    positive = K.pool2d(positive, pool_size=pool_size, padding='same')
    negative = K.pool2d(negative, pool_size=pool_size, padding='same')
    border = positive * negative
    
    return border


def dice_with_edge(y_true, y_pred):
    return dice_coeff_balanced(get_border_mask((5,5), y_true), get_border_mask((5,5), y_pred))


###------------------------------Data Generator----------------------------###
def data_generator(is_train, data):
    while True:
        if is_train:
            np.random.shuffle(data)
        for start in range(0, len(data) // batch_size * batch_size, batch_size):
            x_batch = []
            y_bathc = []
            end = min(start + batch_size, len(data))
            ids_train_batch = data[start:end]
            np.random.shuffle(ids_train_batch)
            for image_id in ids_train_batch:
                #load image
                im = cv2.imread(image_id)
                mask = cv2.imread(image_id.replace('img', 'mask'), 0)
                
                if is_train:
                    '''im,mask = randomShiftScaleRotate(im, mask, 
                                                        shift_limit=(-0.1,0.1), 
                                                        scale_limit=(-0.2,0.2),
                                                        rotate_limit=(-0.1,0.1))
                    im,mask = randomHorzontalFlip(im, mask)'''
                    # im = randomGAMMA(im, 0.5)
                    im = randomBrightness(im, gamma=0.5)
                    # im = randomMotionBlur(im)
                    # im = randomBrightness(im, gamma=2)
                    im = randomContrast(im, gain=1)
                    # im = randomAddNoise2(im)
                    im = randomAddNoise(im)
                    
                try:
                    im = np.reshape(im, (640,1024,3))
                except:
                    print('missing', image_id)
                    
                masks_ = np.zeros((640,1024,2))
                masks_[mask == 128, 0] = 1
                masks_[mask == 255, 1] = 1
                x_batch.append(im)
                y_batch.append(masks_)
                
            x_batch = np.array(x_batch, np.float32) / 255.0
            y_batch = np.array(y_batch, np.uint8)
            yield x_batch, [y_batch, y_batch, y_batch, y_batch, y_batch, y_batch, y_batch, y_batch, y_batch]
            
            
if __name__ == '__main__':
    num_epochs = 1000
    batch_size = 4
    learning_rate = 0.0001
    phase = 'Test'
    
    train_data_dir = r''
    valid_data_dir = r''
    test_dir = r''
    load_weights = r''
    
    train_dir = get_dir(train_data_dir)
    valid_dir = get_dir(valid_data_dir)
    ids_train_split, _ = train_test_split(train_dir, test_size=1, random_state=0)
    ids_valid_split, _ = train_test_split(valid_dir, test_size=1, random_state=0)
    
    input_shape = [640, 1024, 3]
    model = vnet_fpn_relu_SIMO_2(input_size=input_shape)
    
    if len(load_weights) > 0:
        model.load_weights(load_weights, by_name=True)
        
    if phase == 'Train':
        print('Training on {} samples'.format(len(ids_train_split)))
        print('Validating on {} samples'.format(len(ids_valid_split)))
        optimizer = optim.adam(lr=learning_tate, decay=1e-5)
        
        model.compile(optimizer=optimizer, 
                      loss={'output_S1': dice_coeff_balanced, 'output_S2': dice_coeff_balanced, 'output_S3': dice_coeff_balanced, 
                            'output_S4': dice_with_edge, 'output_P1': dice_coeff_balanced, 'output_P2': dice_coeff_balanced, 
                            'output_P3': dice_coeff_balanced, 'output_P4': dice_coeff_balanced, 'output_Fusion': dice_coeff_balanced})
        
        callbacks = [ReduceLROnPlateau(monitor='loss',
                                       factor=0.8,
                                       patience=10,
                                       verbbose=1,
                                       min_delta=1e-4,
                                       mode='min'),
                     ModelCheckpoint(monitor='val_loss',
                                     filepath='',
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='auto'),
                     TensorBoard(log_dir='./log')]
        
        model.fit_generator(generator=data_generator(True, ids_train_split),
                            steps_per_epoch=np.ceil(float(len(ids_train_split))/ float(batch_size)),
                            epochs=num_epochs, verbose=1,
                            callbacks=callbacks,
                            validation_data=data_generator(False, ids_valid_split),
                            validation_steps=np.ceil(float(len(ids_valid_split))/ float(batch_size)))
        
        model.save('')
        
    if phase == 'Test':
        test_pics = glob.glob(os.path.join(test_dir, '*.png'))
        for pic in test_pics:
            print('processing:', pic)
            im = cv2.imread(pic)
            im2 = cv2.imread(pic, 1)
            im = (im / 255.0).astype('float32')
            im_t = np.reshape(im, (1,640,1024,3))
            pred = model.predict(im_t)
            pred2 = np.reshape(pred, (640,1024,2))
            pred2[pred2 > 0.1] = 1
            pred2[pred2 <= 0.1] = 0
            pred2 = np.max(pred2, axis=-1)
            
            pred_ = np.reshape(pred, (640,1024,2)) 
            pred_ = np.argmax(pred_, axis=-1) + 1
            pred_ = pred2 * pred_
            pred_ = pred_.astype('uint8')
            
            pred_bool = pred_.astype(np.bool)
            pred_mask = morphology.remove_small_objects(pred_bool, min_size=10000, connectivity=2)
            pred_ = (pred_ * pred_mask).astype('uint8')
            pred_ = pred_ * 100
            
            pred_sp = pred_.copy()
            pred_sp[pred_sp > 100] = 0
            pred_sp[pred_sp ==100] = 1
            pred_sp = pred_sp.astype(np.bool)
            pred_sp_mask = morphology.remove_small_objects(pred_sp, min_size=500, connectivity=2)
            pred_sp_2 = (pred_sp * pred_sp_mask).astype('uint8')
            for i in range(pred_sp.shape[0]):
                for j in range(pred_sp.shape[1]):
                    if pred_sp[i,j]-pred_sp_2[i,j] != 0:
                        pred_[i,j] = 200
                    else:
                        pass
            
            connectivities = measure.label(pred_sp_2, connectivity=2)
            length = []
            connectivity_mask = pred_.copy()
            for i in range(1, np.max(connectivities)):
                index_list = []
                connectivity_mask[connectivity_mask > 0] = 0
                connectivity_mask[connectivity == i ] = 1
                for j in range(connectivity_mask.shape[1]):
                    if np.sum(connectivity_mask[:,j]) > 0:
                        index_list.append(j)
                if len(index_list)>1:
                    length.append(len(index_list))
                    
            if len(length)>2 and np.max(length)<205:
                pred_[pred_ == 100] = 200
                
            cv2.imwrite(pic.replace(img, 'pred'), pred_)
            
            thresh_1, pred_1 = cv2.threshold(pred_, 50, 255, cv2.THRESH_BINARY)
            _, cnts1, hierarchy1 = cv2.findContours(pred_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts1) > 0:
                cv2.drawContours(im2, cnts1, -1, color=(0,255,0), thickness=1)
                
            thresh_temp, pred_temp = cv2.threshold(pred_, 150, 255, cv2.THRESH_TOZERO_INV)
            thresh_2, pred_2 = cv2.threshold(pred_temp, 50, 255, cv2.THRESH_BINARY)
            __, cnts2, hierarchy2 = cv2.findContours(pred_2, cv2.RETR_EXTERNAL(), cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts2) > 0:
                cv2.drawContours(im2, cnts2, -1, color=(0,0,255), thickness=1)
        
    
            
    
    
    
    
    
    
