from tensorflow.keras import layers
from tensorflow.keras import models

from Blocks import conv_block, enc_block, dec_block, Score

def FNet(base_feature=4, img_shape=(160,160,128)):
    score_layers=[None] * (img_shape[-1])

    inputs = layers.Input(shape=img_shape)
    
    encoder0_pool, encoder0 = enc_block(base_feature)(inputs[...,0:1])
    encoder1_pool, encoder1 = enc_block(2*base_feature)(encoder0_pool)
    encoder2_pool, encoder2 = enc_block(3*base_feature)(encoder1_pool)
    encoder3_pool, encoder3 = enc_block(4*base_feature)(encoder2_pool)
    encoder4_pool, encoder4 = enc_block(5*base_feature)(encoder3_pool)
    center = conv_block(6*base_feature)(encoder4_pool)

    decoder4 = dec_block(5*base_feature)([center, encoder4])
    decoder3 = dec_block(4*base_feature)([decoder4, encoder3])
    decoder2 = dec_block(3*base_feature)([decoder3, encoder2])
    decoder1 = dec_block(2*base_feature)([decoder2, encoder1])
    decoder0 = dec_block(base_feature)([decoder1, encoder0])

    decoder0 = layers.BatchNormalization()(decoder0)
    score_layers[0] = layers.Conv2D(1, (1, 1), activation='relu', padding='same')(decoder0)
    
    score_cluster=Score(base_feature)
    score_bkg=Score(base_feature)
    
    score_layers[1]=score_bkg([inputs[...,1:2],decoder0, decoder1, decoder2, decoder3, decoder4])
    
    for i in range(2,img_shape[-1]):
        score_layers[i] = score_cluster([inputs[...,i:i+1], decoder0, decoder1, decoder2, decoder3, decoder4])
        
    comp_layer = layers.concatenate(score_layers, axis=-1)
    comp_layer = conv_block(img_shape[-1])(comp_layer)
    comp_layer = layers.Conv2D(img_shape[-1], (1, 1), padding='same')(comp_layer)
    outputs = layers.PReLU()(comp_layer)
    
    model = models.Model(inputs=[inputs], outputs=[outputs])
    
    return model
