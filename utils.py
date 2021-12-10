from skimage import data, exposure, img_as_float
import cv2
import numpy as np
import glob
import os


def get_dir(path):
    candidates = glob.glob(os.paht.join(path, '*'+ '.png'))
    
    return candidates


def padding_img(img):
    shape = img.shape
    if shape[0] == shape[1]:
        return cv2.resize(img, (input_size, input_size))
    if shape[0] > shape[1]:
        blk = np.zeros((shape[0], shape[0]))
        blk[:, int((shape[0]-shape[1])/2):int((shape[0]+shape[1])/2)] = img
        return cv2.resize(blk, (input_size, input_size))
    else:
        blk = np.zeros((shape[1],shape[1]))
        blk[int((shape[1]-shape[0])/2):int((shape[1]+shape[0])/2), :] = img
        return cv2.resize(blk, (input_size, input_size))


def randomHueSaturationValue(image, he_shift_limit=(-180, 180), 
                             sat_shift_limit=(-255, 255), 
                             val_shift_limit=(-255, 255), 
                             u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h,s,v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        
    return image


def randomGAMMA(image, u):
    if np.random.random() < u:
        cons = 1.25 - np.random.random()/2
        gam = exposure.adjust_gamma(image, cons)
        
        return gam
    else:
        return image
    
    
def randomShiftScaleRotate(image, mask, 
                           shift_limit=(-0.0625, 0.0625), 
                           scale_limit=(-0.1, 0.1), 
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           border_mode = cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape
        
    angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
    scale = np.random.uniform(1+scale_limit[0], 1+scale_limit[1])
    aspect = np.random.uniform(1+ aspect_limit[0], 1+aspect_limit[1])
    sx = scale * aspect / (aspect **0.5)
    sy = scale / (aspect **0.5)
    dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
    dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)
    
    cc = np.math.cos(angle / 180 * np.math.pi) * sx
    ss = np.math.sin(angle / 180 * np.mati.pi) * sy
    rotate_matrix = np.array([[cc,-ss],[ss,cc]])
    
    box0 = np.array([[0,0], [width,0], [width, height], [0,height],])
    box1 = box0 - np.array([width / 2, height /2])
    box1 = np.dot(box1, rotate_matrix.T) + np.array([width/2 + dx, height/2 + dy])
    
    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)
    image = cv2.warpPerspective(image, mat, (width,height), flags=cv2.INTER_LINEAR, borderMode=border_mode, borderValue=(0,0,0))
    mask = cv2.warpPerspective(mask, mat, (width,height), flags=cv2.INTER_LINEAR, borderMode=border_mode, borderValue=(0,0,0))
    image = image.astype('uint8')
    mask = mask.astype('uint8')
    
    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        
    image = image.astype('uint8')
    mask = mask.astype('uint8')
    
    return image, mask


def generate_all_masks(input_size, num_class, mask):
    all_masks = np.zeros((input_size, input_size, num_class)).astype('uint8')
    all_masks[mask == 0,0] = 1
    all_masks[mask > 0,1] = 1
    
    return all_masks

        
        
        
        
        
        
        