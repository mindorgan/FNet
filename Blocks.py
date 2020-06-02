from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import backend as K

class conv_block(layers.Layer):
    def __init__(self, num_filters, kernel_size=(3,3), activation='ReLU'):
        super(conv_block, self).__init__()
        self.num_filters=num_filters
        self.kernel_size = kernel_size
        self.activation = activation
        
        self.conv_layer1 =  layers.Conv2D(num_filters, kernel_size, padding='same')
        self.batchnorm_layer1 = layers.BatchNormalization()
        if activation.lower() == 'prelu':
            self.activation_layer1 =  layers.PReLU()
        else:
            self.activation_layer1 =  layers.Activation('relu')
        
        self.conv_layer2 =  layers.Conv2D(num_filters, kernel_size, padding='same')
        self.batchnorm_layer2 = layers.BatchNormalization()
        
        if activation.lower() == 'prelu':
            self.activation_layer2 =  layers.PReLU()
        else:
            self.activation_layer2 =  layers.Activation('relu')

    def build(self, input_shape):
        super(conv_block, self).build(input_shape)
        
        num_filters=self.num_filters
        
        self.conv_layer1.build(input_shape)
        self._trainable_weights += self.conv_layer1.trainable_weights
        
        input_shape2=(input_shape[0],input_shape[1],input_shape[2],num_filters)
        
        self.batchnorm_layer1.build(input_shape2)
        self._trainable_weights += self.batchnorm_layer1.trainable_weights
        self.activation_layer1.build(input_shape2)
        self._trainable_weights += self.activation_layer1.trainable_weights
        
        self.conv_layer2.build(input_shape2)
        self._trainable_weights += self.conv_layer2.trainable_weights
        self.batchnorm_layer2.build(input_shape2)
        self._trainable_weights += self.batchnorm_layer2.trainable_weights
        self.activation_layer2.build(input_shape2)
        self._trainable_weights += self.activation_layer2.trainable_weights
        
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'activation': self.activation,
            'conv_layer1': self.conv_layer1,
            'batchnorm_layer1': self.batchnorm_layer1,
            'activation_layer1': self.activation_layer1,
            'conv_layer2': self.conv_layer2,
            'batchnorm_layer2': self.batchnorm_layer2,
            'activation_layer2': self.activation_layer2
        })
        return config
    
    def call(self, x):
        x = self.conv_layer1(x)
        x = self.batchnorm_layer1(x)
        x = self.activation_layer1(x)
        
        x = self.conv_layer2(x)
        x = self.batchnorm_layer2(x)
        x = self.activation_layer2(x)
        return x
    
class enc_block(layers.Layer):
    def __init__(self, num_filters, kernel_size=(3,3), activation='ReLU'):
        super(enc_block, self).__init__()
        
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.activation = activation
        
        self.conv_layer = conv_block(num_filters, kernel_size, activation)
        self.pool_layer = layers.MaxPooling2D((2, 2), strides=(2, 2))

    def build(self, input_shape):
        super(enc_block, self).build(input_shape)
        
        num_filters=self.num_filters
        if len(input_shape) < 4:
            input_shape=(input_shape[0],input_shape[1],input_shape[2],1)
            
        self.conv_layer.build(input_shape)
        self._trainable_weights += self.conv_layer.trainable_weights
        
        input_shape2=(input_shape[0],input_shape[1],input_shape[2],num_filters)
        
        self.pool_layer.build(input_shape2)
        self._trainable_weights += self.pool_layer.trainable_weights
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'activation': self.activation,
            'conv_layer': self.conv_layer,
            'pool_layer': self.pool_layer
        })
        return config
    
    def call(self, x):
        if len(x.shape)<4:
            x=K.expand_dims(x,axis=-1)
        x = self.conv_layer(x)
        return self.pool_layer(x), x
    
class dec_block(layers.Layer):
    def __init__(self, num_filters, kernel_size=(3,3), activation='ReLU'):
        super(dec_block, self).__init__()
        self.num_filters=num_filters
        self.kernel_size=kernel_size
        self.activation = activation
        self.conv_layer =  conv_block(num_filters, kernel_size, activation)
        
    def build(self, input_shape0):
        super(dec_block, self).build(input_shape0)
        
        num_filters=self.num_filters
        kernel_size=self.kernel_size
        
        input_shape=input_shape0[0]
        concat_shape=input_shape0[1]
        
        merge_shape=(concat_shape[0],concat_shape[1],concat_shape[2],num_filters+concat_shape[3])
        
        #self._trainable_weights=[]
        if input_shape[1]<concat_shape[1]:
            self.convT_layer =  layers.Conv2DTranspose(num_filters, kernel_size, strides=(2, 2), padding='same')
        else:
            self.convT_layer =  layers.Conv2D(num_filters, kernel_size, padding='same')
        
        self.convT_layer.build(input_shape)
        self._trainable_weights += self.convT_layer.trainable_weights
        
        input_shape2=(concat_shape[0],concat_shape[1],concat_shape[2],num_filters)
        
        self.conv_layer.build(merge_shape)
        self._trainable_weights += self.conv_layer.trainable_weights
        
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'activation': self.activation,
            'convT_layer': self.convT_layer,
            'conv_layer': self.conv_layer
        })
        return config
    
    def call(self, inputs):
        x=inputs[0]
        concat_tensor=inputs[1]
        
        input_shape=x.shape
        concat_shape=concat_tensor.shape
        
        x = self.convT_layer(x)
        x = layers.concatenate([concat_tensor, x],axis=-1)
        
        x = self.conv_layer(x)
        return x
    
class Score(layers.Layer):
    def __init__(self, base_feature=4, kernel_size=(3,3), activation='ReLU'):
        super(Score, self).__init__()
        self.base_feature=base_feature
        self.kernel_size=kernel_size
        self.activation=activation

    def build(self, input_shape0):
        super(Score, self).build(input_shape0)
        
        base_feature=self.base_feature
        kernel_size = self.kernel_size
        activation = self.activation
        
        input_shape=input_shape0[0]
        dec0_shape=input_shape0[1]
        dec1_shape=input_shape0[2]
        dec2_shape=input_shape0[3]
        dec3_shape=input_shape0[4]
        dec4_shape=input_shape0[5]

        self.encoder_block0 = enc_block(base_feature, kernel_size, activation)
        self.encoder_block0.build(input_shape)
        self._trainable_weights += self.encoder_block0.trainable_weights

        self.encoder_block1 = enc_block(2*base_feature, kernel_size, activation)
        self.encoder_block1.build((input_shape[0],input_shape[1]//2,input_shape[2]//2,base_feature))
        self._trainable_weights += self.encoder_block1.trainable_weights

        self.encoder_block2 = enc_block(3*base_feature, kernel_size, activation)
        self.encoder_block2.build((input_shape[0],input_shape[1]//4,input_shape[2]//4,2*base_feature))
        self._trainable_weights += self.encoder_block2.trainable_weights

        self.encoder_block3 = enc_block(4*base_feature, kernel_size, activation)
        self.encoder_block3.build((input_shape[0],input_shape[1]//8,input_shape[2]//8,3*base_feature))
        self._trainable_weights += self.encoder_block3.trainable_weights

        self.encoder_block4 = enc_block(5*base_feature, kernel_size, activation)
        self.encoder_block4.build((input_shape[0],input_shape[1]//16,input_shape[2]//16,4*base_feature))
        self._trainable_weights += self.encoder_block4.trainable_weights

        self.center = conv_block(6*base_feature, kernel_size, activation)
        self.center.build((input_shape[0],input_shape[1]//32,input_shape[2]//32,5*base_feature))
        self._trainable_weights += self.center.trainable_weights

        self.decoder_block4 = dec_block(6*base_feature, kernel_size, activation)
        shape1=(input_shape[0],input_shape[1]//32,input_shape[2]//32,6*base_feature)
        shape2=(input_shape[0],input_shape[1]//16,input_shape[2]//16,5*base_feature+dec4_shape[3])
        self.decoder_block4.build([shape1, shape2])
        self._trainable_weights += self.decoder_block4.trainable_weights

        self.decoder_block3 = dec_block(5*base_feature, kernel_size, activation)
        shape1=(input_shape[0],input_shape[1]//16,input_shape[2]//16,6*base_feature)
        shape2=(input_shape[0],input_shape[1]//8,input_shape[2]//8,4*base_feature+dec3_shape[3])
        self.decoder_block3.build([shape1, shape2])
        self._trainable_weights += self.decoder_block3.trainable_weights

        self.decoder_block2 = dec_block(4*base_feature, kernel_size, activation)
        shape1=(input_shape[0],input_shape[1]//8,input_shape[2]//8,5*base_feature)
        shape2=(input_shape[0],input_shape[1]//4,input_shape[2]//4,3*base_feature+dec2_shape[3])
        self.decoder_block2.build([shape1, shape2])
        self._trainable_weights += self.decoder_block2.trainable_weights

        self.decoder_block1 = dec_block(3*base_feature, kernel_size, activation)
        shape1=(input_shape[0],input_shape[1]//4,input_shape[2]//4,4*base_feature)
        shape2=(input_shape[0],input_shape[1]//2,input_shape[2]//2,2*base_feature+dec1_shape[3])
        self.decoder_block1.build([shape1, shape2])
        self._trainable_weights += self.decoder_block1.trainable_weights

        self.decoder_block0 = dec_block(2*base_feature, kernel_size, activation)
        shape1=(input_shape[0],input_shape[1]//2,input_shape[2]//2,3*base_feature)
        shape2=(input_shape[0],input_shape[1],input_shape[2],base_feature+dec0_shape[3])
        self.decoder_block0.build([shape1, shape2])
        self._trainable_weights += self.decoder_block0.trainable_weights

        self.conv_block = conv_block(base_feature, kernel_size, activation)
        self.conv_block.build((input_shape[0],input_shape[1],input_shape[2],2*base_feature))
        self._trainable_weights += self.conv_block.trainable_weights

        self.batchnorm = layers.BatchNormalization()
        self.batchnorm.build((input_shape[0],input_shape[1],input_shape[2],base_feature))

        self.score_layer = layers.Conv2D(1, (1, 1), activation='relu', padding='same')
        self.score_layer.build((input_shape[0],input_shape[1],input_shape[2],base_feature))
        self._trainable_weights += self.score_layer.trainable_weights

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'base_feature': self.base_feature,
            'kernel_size': self.kernel_size,
            'activation': self.activation,
            'encoder_block0': self.encoder_block0,
            'encoder_block1': self.encoder_block1,
            'encoder_block2': self.encoder_block2,
            'encoder_block3': self.encoder_block3,
            'encoder_block4': self.encoder_block4,
            'center': self.center,
            'decoder_block4': self.decoder_block4,
            'decoder_block3': self.decoder_block3,
            'decoder_block2': self.decoder_block2,
            'decoder_block1': self.decoder_block1,
            'decoder_block0': self.decoder_block0,
            'conv_block': self.conv_block,
            'batchnorm': self.batchnorm,
            'score_layer': self.score_layer
        })
        return config
    def call(self, inputs0):
        base_feature=self.base_feature

        inputs=inputs0[0]
        decoder0=inputs0[1]
        decoder1=inputs0[2]
        decoder2=inputs0[3]
        decoder3=inputs0[4]
        decoder4=inputs0[5]

        encoder0_pool_cl, encoder0_cl = self.encoder_block0(inputs)
        encoder1_pool_cl, encoder1_cl = self.encoder_block1(encoder0_pool_cl)
        encoder2_pool_cl, encoder2_cl = self.encoder_block2(encoder1_pool_cl)
        encoder3_pool_cl, encoder3_cl = self.encoder_block3(encoder2_pool_cl)
        encoder4_pool_cl, encoder4_cl = self.encoder_block4(encoder3_pool_cl)
        center = self.center(encoder4_pool_cl)
        decoder4_cl = self.decoder_block4([center, layers.concatenate([decoder4, encoder4_cl], axis=-1)])
        decoder3_cl = self.decoder_block3([decoder4_cl, layers.concatenate([decoder3, encoder3_cl], axis=-1)])
        decoder2_cl = self.decoder_block2([decoder3_cl, layers.concatenate([decoder2, encoder2_cl], axis=-1)])
        decoder1_cl = self.decoder_block1([decoder2_cl, layers.concatenate([decoder1, encoder1_cl], axis=-1)])
        decoder0_cl = self.decoder_block0([decoder1_cl, layers.concatenate([decoder0, encoder0_cl], axis=-1)])

        decoder0_cl_f = self.conv_block(decoder0_cl)
        decoder0_cl_f = self.batchnorm(decoder0_cl_f)
        return self.score_layer(decoder0_cl_f)