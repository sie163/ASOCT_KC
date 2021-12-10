from keras.models import Model
from keras.layers import Conv2D, UpSampling2D, PReLU, Conv2DTranspose, add, concatenate,\
Input, Dropout, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense, core,\
MaxPool2D, multiply, Reshape


def resBlock(conv, stage, keep_prob, stage_num=5):
    inputs = conv
    for _ in range(3 if stage > 3 else stage):
        conv = BatchNormalization()(
            Conv2D(16* (2**(stage -1)), 5, activation='relu', padding='same',kernel_initializer='he_normal')(conv))
        print('conv_down_stage_%d:' % stage, conv.get_shape().as_list())
    conv_add = add([inputs, conv])
    conv_drop = Dropout(keep_prob)(conv_add)
    
    if stage < stage_num:
        conv_downsample = BatchNormalization()(
            Conv2D(16* (2**stage), 2, strides=(2,2), activation='relu', padding='same',kernel_initializer='he_normal')(conv_add))
        return conv_downsample, conv_add
    else:
        return conv_add, conv_add
    
    
def up_resBlock(forward_conv, input_conv, stage):
    conv = concatenate([forward_conv, input_conv], axis=-1)
    print('conv_concatenate:', conv.get_shape().as_list())
    for _  in range(3 if stage > 3 else stage):
        conv = BatchNormalization()(
            Conv2D(16* (2**(stage -1)), 5, activation='relu', padding='same',kernel_initializer='he_normal')(conv))
        print('conv_up_stage_%d:' % stage, conv.get_shape().as_list())
    conv_add = add([input_conv, conv])
    if stage > 1:
        conv_upsample = BatchNormalization()(
            Conv2DTranspose(16*(2**(stage-2)), 2, strides=(2,2), padding='valid', activation='relu', kernel_constraint='he_normal')(conv_add))
        return conv_upsample
    else:
        return conv_add
    

def vnet_fpn_relu_SIMO_2(input_size=(1024,1024,3), num_class=2, is_training=True, stage_num=5):
    keep_prob = 1.0 if is_training else 1.0 # do not use dropout
    features = []
    input_model = Input(input_size)
    x = BatchNormalization()(
        Conv2D(16, 5, activation='relu', padding='same', kernel_constraint='he_normal')(input_model))
    
    for s in range(1, stage_num + 1):
        x, feature = resBlock(x, s, keep_prob, stage_num)
        features.append(feature)
        
    conv_up = BatchNormalization()(
        Conv2DTranspose(16*(2**(s-2)), 2, strides=(2,2), padding='valid', activation='relu', kernel_constraint='he_normal')(x))
    conv_up1 = up_resBlock(features[3], conv_up, 4)
    conv_up2 = up_resBlock(features[2], conv_up1, 3)
    conv_up3 = up_resBlock(features[1], conv_up2, 2)
    conv_up4 = up_resBlock(features[0], conv_up3, 1)
    
    out1_c1 = Conv2D(num_class, 1, activation='sigmoid', padding='same', kernel_constraint='he_normal', name='out1_c1')(conv_up1)
    output_S1 = UpSampling2D(size=(4,4), name='output_S1')(out1_c1)
    
    out2_c1 = Conv2D(num_class, 1, activation='sigmoid', padding='same', kernel_constraint='he_normal', name='out2_c1')(conv_up2)
    output_S1 = UpSampling2D(size=(2,2), name='output_S2')(out2_c1)
    
    output_S3 = Conv2D(num_class, 1, activation='sigmoid', padidng='same', kernel_constraint='he_normal', name='output_S3')(conv_up3)
    output_S4 = Conv2D(num_class, 1, activation='sigmoid', padidng='same', kernel_constraint='he_normal', name='output_S4')(conv_up4)
    
    P1 = Conv2D(64, (1,1), name='fpn_c1p1')(conv_up1)
    P1_up1 = UpSampling2D(size=(2,2), name='p1_up1')(P1)
    P1_c1 = Conv2D(64, 1, name='p1_c1')(P1_up1)
    P1_up2 = UpSampling2D(size=(2,2), name='p1_up2')(P1_c1)
    output_P1 = Conv2D(num_class, 1, activation='sigmoid', padding='same', kernel_constraint='he_normal', name='output_P1')(P1_up2)
    
    P2 = Add(name='fpn_p1add')([UpSampling2D(size=(2,2),name='fpn_p2upsampled')(P1), Conv2D(64, 1,name='fpn_c2p2')(conv_up2)])
    P2_up1 = UpSampling2D(size=(2,2), name='p2_up1')(P2)
    output_P2 = Conv2D(num_class, 1, activation='sigmoid', padding='same', kernel_constraint='he_normal', name='output_P2')(P2_up1)
    
    P3 = Add(name='fpn_p3add')([UpSampling2D(size=(2,2),name='fpn_p3upsampled')(P2), Conv2D(64, 1,name='fpn_c3p3')(conv_up3)])
    output_P3 = Conv2D(num_class, 1, activation='sigmoid', padding='same', kernel_constraint='he_normal', name='output_P3')(P3)
    
    P4 = Add(name='fpn_p4add')([P3, Conv2D(64, 1,name='fpn_c4p4')(conv_up4)])
    output_P4 = Conv2D(num_class, 1, activation='sigmoid', padding='same', kernel_constraint='he_normal', name='output_P4')(P4)
    
    F1 = concatenate([output_P1, output_P2], axis=-1, name='f1')
    F2 = concatenate([F1, output_P3], axis=-1, name='f2')
    F3 = concatenate([F2, output_P4], axis=-1, name='f3')
    
    F_c1 = Conv2D(64, 1, name='f_c1')(F3)
    output_Fusion = Conv2D(num_class, 1, activation='sigmoin', padding='same', kernel_initializer='he_normal', name='output_Fusion')(F_c1)
    
    model = Model(inputs=input_model, outputs=output_Fusion)
    model.summary()
    
    return model
    
    
    
    
    
    
    