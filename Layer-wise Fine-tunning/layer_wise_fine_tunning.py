
from keras.optimizers import Adam, RMSprop
from tqdm import tqdm
from keras.utils.layer_utils import count_params

# model, epochs, ordered_layre_name, lr, reduce
# epochs = 50
# lr = 1e-2
# reduce =1
# ordered_layers_name = ["block5_conv3","block4_conv3","block3_conv3","block2_conv2","block1_conv2"]
def layer_wise_training(model= [],
                        loss = 'categorical_crossentropy',
                        metrics =['acc'],
                        epochs =[1], # epochs per layer
                        lr =[1e-1],  # learning_rate per layer
                        callbacks = [],
                        class_weight = [],
                        train_generator = [],
                        test_generator =[],
                        ordered_layers_name =[] # from top(Dense) to bottom(Input)
                       ):
  
  if len(epochs) != len(ordered_layers_name):
    print('Input error!!!')
    
  if (train_generator == []) or (test_generator == []) or (model==[]) :
    print('Generator error!!!')
      
      
   

  print('\nLayerWise Training\n')



#     print(lr)

  initial_epoch = 0
  for idx, layer_name in enumerate(tqdm(ordered_layers_name)):


      # Learning Rate stays same for first epoch
#         if idx == 0 :
#             lr = lr
#         else:
#             lr = lr*reduce


      # UnFreezeing All the layers
      for x in range(len(model.layers)):
        model.layers[x].trainable = True


      # Finding layer index for Unfreeze
      layer_name = layer_name
      index = None
      for x in range(len(model.layers)):
          if model.layers[x].name == layer_name:
              index = x

      # Freezeing the layers
      fine_tune_at = index
      for x in range(fine_tune_at+1):
        model.layers[x].trainable = False
  #       print(model.layers[x], model.layers[x].trainable)


      # Compiling the model
       
      
      model.compile(optimizer= Adam(lr = lr[idx]), loss= loss , metrics= metrics)
#         print(lr*reduce)

      # Training the Model
      if class_weight == []:
            class_weight = np.full(len(np.unique(train_generator.labels)), 1,  dtype = int)
            

            
      print(f'Training Stage: {idx+1}  ||  Total Trainable Parameters: {count_params(model.trainable_weights):,d}  ||  Learning_rate: {lr[idx]:,.7f}')
      print('==================================================================================')
      print('\n')
      model.fit_generator(train_generator,
                          epochs    = initial_epoch + epochs[idx],
                          steps_per_epoch  = train_generator.samples // train_generator.batch_size,
                          validation_data  = test_generator,
                          validation_steps = test_generator.samples // test_generator.batch_size,
                          class_weight = class_weight,
                          callbacks = callbacks,
                          initial_epoch = initial_epoch)

      initial_epoch = initial_epoch + epochs[idx]

  #     print('training')
      print('============================================================================================')
      print('\n')
