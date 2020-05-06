def Network2(Input_shape, dropout = False):

  from keras.layers import Dense, Dropout, Conv2D, BatchNormalization, SeparableConv2D, Concatenate
  from keras.layers import MaxPool2D, Input, Add, DepthwiseConv2D, GlobalAveragePooling2D, Flatten
  from keras.models import Model, load_model
  from keras.utils import plot_model
  from keras.optimizers import Adam, SGD, RMSprop
  
  xin = Input(Input_shape)

  x = Conv2D(16, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu')(xin)
  x = BatchNormalization()(x)

  x1a = SeparableConv2D(16, kernel_size=(3,3), padding='same', activation='relu')(x)
  x1a = BatchNormalization()(x1a)

  x1b = SeparableConv2D(16, kernel_size=(3,3), padding='same', activation='relu')(x1a)
  x1b = BatchNormalization()(x1b)

  x1c = SeparableConv2D(16, kernel_size=(3,3), padding='same', activation='relu')(x1b)
  x1c = BatchNormalization()(x1c)

  x1d = SeparableConv2D(16, kernel_size=(3,3), padding='same', activation='relu')(x1c)
  x1d = BatchNormalization()(x1d)


  x = Concatenate(axis=-1)([x1a, x1b, x1c, x1d])
  x = Conv2D(16, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu')(x)

  x = MaxPool2D()(x)



  x1a = SeparableConv2D(32, kernel_size=(3,3), padding='same', activation='relu')(x)  
  x1a = BatchNormalization()(x1a)

  x1b = SeparableConv2D(32, kernel_size=(3,3), padding='same', activation='relu')(x1a)
  x1b = BatchNormalization()(x1b)

  x1c = SeparableConv2D(32, kernel_size=(3,3), padding='same', activation='relu')(x1b)
  x1c = BatchNormalization()(x1c)

  x1d = SeparableConv2D(32, kernel_size=(3,3), padding='same', activation='relu')(x1c)
  x1d = BatchNormalization()(x1d)


  x = Concatenate(axis=-1)([x1a, x1b, x1c, x1d])
  x = Conv2D(32, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu')(x)

  x = MaxPool2D()(x)


  x1a = SeparableConv2D(64, kernel_size=(3,3), padding='same', activation='relu')(x)
  x1a = BatchNormalization()(x1a)

  x1b = SeparableConv2D(64, kernel_size=(3,3), padding='same', activation='relu')(x1a)
  x1b = BatchNormalization()(x1b)

  x1c = SeparableConv2D(64, kernel_size=(3,3), padding='same', activation='relu')(x1b)
  x1c = BatchNormalization()(x1c)

  x1d = SeparableConv2D(64, kernel_size=(3,3), padding='same', activation='relu')(x1c)
  x1d = BatchNormalization()(x1d)


  x = Concatenate(axis=-1)([x1a, x1b, x1c, x1d])
  if dropout:
    x = Dropout(dropout)(x)
  x = Conv2D(64, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu')(x)

  x = MaxPool2D()(x)

  x1a = SeparableConv2D(128, kernel_size=(3,3), padding='same', activation='relu')(x)
  x1a = BatchNormalization()(x1a)

  x1b = SeparableConv2D(128, kernel_size=(3,3), padding='same', activation='relu')(x1a)
  x1b = BatchNormalization()(x1b)

  x1c = SeparableConv2D(128, kernel_size=(3,3), padding='same', activation='relu')(x1b)
  x1c = BatchNormalization()(x1c)

  x1d = SeparableConv2D(128, kernel_size=(3,3), padding='same', activation='relu')(x1c)
  x1d = BatchNormalization()(x1d)


  x = Concatenate(axis=-1)([x1a, x1b, x1c, x1d])
  if dropout:
    x = Dropout(dropout)(x)
  x = Conv2D(128, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu')(x)

  x = MaxPool2D()(x)

  x1a = SeparableConv2D(256, kernel_size=(3,3), padding='same', activation='relu')(x)
  x   = BatchNormalization()(x1a)

  x1b = SeparableConv2D(256, kernel_size=(3,3), padding='same', activation='relu')(x1a)
  x1b = BatchNormalization()(x1b)

  x1c = SeparableConv2D(256, kernel_size=(3,3), padding='same', activation='relu')(x1b)
  x1c = BatchNormalization()(x1c)

  x1d = SeparableConv2D(256, kernel_size=(3,3), padding='same', activation='relu')(x1c)
  x1d = BatchNormalization()(x1d)


  x = Concatenate(axis=-1)([x1a, x1b, x1c, x1d])
  if dropout:
    x = Dropout(dropout)(x)
  x = Conv2D(256, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu')(x)

  x = MaxPool2D()(x)

  x1a = SeparableConv2D(256, kernel_size=(3,3), padding='same', activation='relu')(x)
  x1a = BatchNormalization()(x1a)

  x1b = SeparableConv2D(256, kernel_size=(3,3), padding='same', activation='relu')(x1a)
  x1b = BatchNormalization()(x1b)

  x1c = SeparableConv2D(256, kernel_size=(3,3), padding='same', activation='relu')(x1b)
  x1c = BatchNormalization()(x1c)


  x = Concatenate(axis=-1)([x1a, x1b, x1c])
  if dropout:
    x = Dropout(dropout)(x)
  x = Conv2D(256, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu')(x)

  x = GlobalAveragePooling2D()(x)

  x = Dense(128, activation='relu')(x)
  
  if dropout:
    x = Dropout(dropout)(x)
    
  x = Dense(64, activation='relu')(x)
  
  if dropout:
    x = Dropout(dropout)(x)

  out = Dense(1, activation='sigmoid')(x)

  model = Model(xin, out)
  
  model.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])
  
  return model
