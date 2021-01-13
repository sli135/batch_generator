#!/usr/bin/python
import numpy as np
import csv
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import batchgenerator as bg

tf.compat.v1.experimental.output_all_intermediates(True)
tf.compat.v1.disable_eager_execution()
tf.__version__

from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, SeparableConv1D,MaxPooling2D, Dropout
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Add, Convolution2D, Conv2D, Conv2DTranspose, multiply, LeakyReLU, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from functools import partial
from tensorflow.keras.layers import Lambda

BATCH_SIZE = 20

# The training ratio is the number of discriminator updates
# per generator update. The paper uses 5.
TRAINING_RATIO = 5
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper
time = 350

numEpochs = None
trainEnergy = True
def reweight(label,types):
    graph_path = './pics'
    n_evts_target = 10000
    # Histogram the data into e_charge bins, then re-weight data within each bin
    ec_min=min(label)
    ec_max=max(label)
    no_outlier = False
    bins = 50
    if types == "charge_e":
        x_label = types + " [keV]"
    else:
        x_label = types + " [cm]"
    while not no_outlier:
        ec_bins = np.linspace(ec_min, ec_max, bins)
        print(ec_max,ec_min)
        (er_area, er_bins, _) = plt.hist(label,bins=ec_bins, label=types, color='r')
        ec_ind_er = np.digitize(label, bins=ec_bins)-1 # have to subtract 1 due to bin indexing
        if bins - 1 in ec_ind_er:
            ec_max += (ec_max - ec_min) / bins
        else:
            no_outlier = True
    plt.xlabel(x_label)
    plt.legend()
    plt.title('Before re-weighting')
    plt.savefig(graph_path + "/%s_before_re-weighting_850.png" % types,bbox_inches = 'tight')
    plt.show()
    # Get weights
    weight_scale = n_evts_target/np.sum(1/er_area[ec_ind_er]) # scale weights to keep them of order 1
    weight_er = weight_scale/er_area[ec_ind_er]

    plt.figure()
    plt.hist(label,bins=ec_bins,weights=weight_er, label=types, color='r')
    plt.xlabel(x_label)
    plt.legend()
    plt.title('After re-weighting')
    plt.savefig(graph_path + "/%s_after_re-weighting_850.png" % types,bbox_inches = 'tight')
    plt.show()


    print("Total DD events after cuts: ",len(label))

    print("Total DD weight after cuts: ",np.sum(weight_er))
    return weight_er
class RandomWeightedAverage(Add):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.

    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

def make_auxiliary_classifier(label,noise):
    l0 = concatenate([noise,label],axis = 1)
    l1 = Dense(10)(l0)
    l2 = Dense(100)(l1)
    #l3 = Reshape((1,100,1))(l2)
    return l2
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight):
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors 
    ## tf.gradients is not supported when eager execution is enabled. Use tf.GradientTape instead.
    gradients = K.gradients(y_pred, averaged_samples)[0]
    print('gradients',gradients)
    #with tf.GradientTape() as g:
    #    g.watch(averaged_samples)
    #    gradients = g.gradient(y_pred, averaged_samples)
    print('g.gradient',gradients)
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)
def constrainer_loss(y_true, y_pred):
    k = 1
    return tf.reduce_mean((y_true - y_pred) ** 2) 

def DenselyConnectedSepConv(z, nfilter, **kwargs):
    ''' Densely Connected SeparableConvolution2D Layer'''
    c = SeparableConv1D(nfilter, 3, padding = 'same', depth_multiplier=1, **kwargs)(z)
    return concatenate([z, c], axis=-1)

def make_constrainer(train_energy):
    print(train_energy)
    input_shape = (74, 350, 1)
    model = Sequential()
    model.add(Conv2D(16, (5, 5), padding = 'same',input_shape = input_shape,
        kernel_regularizer=l2(5e-3)))
    model.add(MaxPooling2D(pool_size=(2, 3), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(32, (5, 5), padding = 'same',
        kernel_regularizer=l2(5e-3)))
    model.add(MaxPooling2D(pool_size=(2, 3), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(64, (5, 5), padding = 'same',
        kernel_regularizer=l2(5e-3)))
    model.add(MaxPooling2D(pool_size=(2, 4), strides=None, padding='valid', data_format=None))
    model.add(Conv2D(128, (5, 5), padding = 'same',
        kernel_regularizer=l2(5e-3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=None, padding='valid', data_format=None))
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Dense(1024))
    model.add(Dense(256))
    if train_energy == True:
        model.add(Dense(1))
    else:
        model.add(Dense(3))
    opt_c = Adam(2e-6,beta_1 = 0.5,beta_2 = 0.9, decay= 0.0)
    model.compile(optimizer = opt_c,loss = 'mean_squared_error')
    model.summary()
    return model
def make_discriminator():
    input_shape = (74, 350, 1)
    model = Sequential()
    model.add(Conv2D(16, (5, 5), padding = 'same', input_shape = input_shape)) # , input_shape = (74,100,1)
    model.add(LeakyReLU())
    model.add(Conv2D(64, (5, 5), kernel_initializer='he_normal', padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(64, (5, 5), kernel_initializer='he_normal', padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(64, (5, 5), kernel_initializer='he_normal', padding='same'))
    model.add(LeakyReLU())
    ########### Addtional layers ########################
    
    model.add(Conv2D(64, (5, 5), kernel_initializer='he_normal', padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(64, (5, 5), kernel_initializer='he_normal', padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(64, (5, 5), kernel_initializer='he_normal', padding='same'))
    model.add(LeakyReLU())
    
    #####################################################
    model.add(Flatten())
    model.add(Dense(1,kernel_initializer='he_normal',name = 'discriminator_output'))
    model.summary()
    
    return model
def regression_loss(y_true,y_pred,
                          label,k):
    #k = 0.01
    print('y_pred',y_pred.shape)
    print('label',label.shape)
    print(y_pred[:BATCH_SIZE][1].shape)
    print(y_pred[BATCH_SIZE:][1].shape)
    e_real = tf.reduce_mean((label - y_pred[:BATCH_SIZE]) ** 2)
    e_gen = tf.reduce_mean((label - y_pred[BATCH_SIZE:]) ** 2)
    return k * K.abs(e_real - e_gen)
def make_generator():
    model = Sequential()
    model.add(Dense(5*11*16, input_dim=100))
    model.add(LeakyReLU())
    model.add(Reshape((5,11,16), input_shape=(5*11*16,)))
    model.add(Conv2DTranspose(16,(3,3), strides = (2,4) ,padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(32,(3,3), strides = 2 ,padding='same'))
    #model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(64,(3,3), strides = 2 ,padding='same'))
    #model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(128,(3,3), strides = 2 ,padding='same'))
    #model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2D(128,(7,3), padding = 'valid'))
    #model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2D(128, (5, 5), padding='same')) #change in Conv2D and no strides needed
    #model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2D(128, (5, 5), padding='same'))
    #model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2D(128, (5, 5), padding='same')) #change in Conv2D and no strides needed
    #model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(1, (1,1), padding='same'))
    model.summary()
    noise = Input(shape=(100,),name = 'noise')
    label = Input(shape=(4,), name = 'label')
    model_input = make_auxiliary_classifier(label,noise)
    
    image = model(model_input)
    return Model(inputs = [label,noise], outputs = [image])
def save_models(save_model_path,name,generator,discriminator,constrainer_e,constrainer_p):
    generator.save(save_model_path + 'model_generator_'+name+'.h5')
    discriminator.save(save_model_path + 'model_discriminator_'+name+'.h5')
    constrainer_e.save(save_model_path + 'model_constrainer_e_'+name+'.h5')#,include_optimizer = False)
    constrainer_p.save(save_model_path + 'model_constrainer_p_'+name+'.h5')#,include_optimizer = False)

def save_weights(save_model_path,name,generator_model,discriminator_model,constrainer_e,constrainer_p):
    symbolic_weights = getattr(generator_model.optimizer, 'weights')
    weight_values = K.batch_get_value(symbolic_weights)
    with open(save_model_path + 'model_generator_weights_'+name+'.pkl', 'wb') as f:
        pickle.dump(weight_values, f,pickle.HIGHEST_PROTOCOL)
    symbolic_weights = getattr(discriminator_model.optimizer, 'weights')
    weight_values = K.batch_get_value(symbolic_weights)
    with open(save_model_path + 'model_discriminator_weights_'+name+'.pkl', 'wb') as f:
        pickle.dump(weight_values, f,pickle.HIGHEST_PROTOCOL)
    symbolic_weights = getattr(constrainer_e.optimizer, 'weights')
    weight_values = K.batch_get_value(symbolic_weights)
    with open(save_model_path + 'model_constrainer_e_weights_'+name+'.pkl', 'wb') as f:
        pickle.dump(weight_values, f,pickle.HIGHEST_PROTOCOL)
    symbolic_weights = getattr(constrainer_p.optimizer, 'weights')
    weight_values = K.batch_get_value(symbolic_weights)
    with open(save_model_path + 'model_constrainer_p_weights_'+name+'.pkl', 'wb') as f:
        pickle.dump(weight_values, f,pickle.HIGHEST_PROTOCOL)
def save_loss(critic,dis,con_e,con_p,epoch):
    csv_path = '/gpfs/slac/staas/fs1/g/g.exo/shaolei/GAN_loss/'
    with open(csv_path + 'critic_loss_constrain_'+str(epoch)+'.csv', mode='w') as csv_file:
        fieldnames = ['loss', 'wasserstein1', 'wasserstein2','gradient_penalty']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for row in critic:
            writer.writerow({'loss': row[0], 'wasserstein1': row[1], 'wasserstein2': row[2], 'gradient_penalty':row[3]})
    with open(csv_path + 'generator_loss_constrain_'+str(epoch)+'.csv', mode='w') as csv_file:
        fieldnames = ['loss', 'discriminator_output_loss', 'constrainer_e_loss','constrainer_p_loss']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for row in dis:
            writer.writerow({'loss': row[0], 'discriminator_output_loss': row[1], 'constrainer_e_loss': row[2], 'constrainer_p_loss':row[3]})
    with open(csv_path + 'constrainer_e_loss_constrain_'+str(epoch)+'.csv', mode='w') as csv_file:
        fieldnames = ['loss']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for row in con_e:
            writer.writerow({'loss': row})
    with open(csv_path + 'constrainer_p_loss_constrain_'+str(epoch)+'.csv', mode='w') as csv_file:
        fieldnames = ['loss']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for row in con_p:
            writer.writerow({'loss': row})
def read_data(filename):
    nLabels = 4
    nFeatures = 74*350

    features = []
    labels = []
    count = 0
    with open(filename) as infile:
        for line in infile:
            x = np.fromstring(line, dtype=float, sep=',')
            features.append(x[0:nFeatures])
            labels.append(x[nFeatures:nFeatures+4])

            count += 1
    print(filename + " has %i events" % count)
    return np.asarray(features), np.asarray(labels)
def train(Epochs):
    print('Start training. Epochs:',Epochs)
    data_path = "/gpfs/slac/staas/fs1/g/g.exo-userdata/users/shaolei/"
    save_model_path = '/gpfs/slac/staas/fs1/g/g.exo/shaolei/GAN_reweight_models/'
    ################ batch generator ##############################
    X_train,Y_train = bg.load_data("train_files/filenames_train.npy","train_files/labels_train.npy")
    X_e_train,Y_e_train = bg.load_data("train_files/filenames_train.npy","train_files/labels_e_train.npy")
    X_p_train,Y_p_train = bg.load_data("train_files/filenames_train.npy","train_files/labels_p_train.npy")
    train_batch_generator = bg.Batch_Generator(X_train,Y_train, BATCH_SIZE)
    train_e_batch_generator = bg.Batch_Generator(X_e_train,Y_e_train, BATCH_SIZE)
    train_p_batch_generator = bg.Batch_Generator(X_p_train,Y_p_train, BATCH_SIZE)
    #val_e_batch_generator = Batch_Generator("filenames_test.npy","labels_e_test.npy", BATCH_SIZE)
    #val_p_batch_generator = Batch_Generator("filenames_test.npy","labels_p_test.npy", BATCH_SIZE)
    ############Make constrainers ########################
    constrainer_p = make_constrainer(train_energy = False)
    constrainer_e = make_constrainer(train_energy = True)

    ############Pre train constrainers ########################

    constrainer_e.fit_generator(generator=train_e_batch_generator,epochs = 3,verbose = 1) #sample_weight=e_charge_weight
    constrainer_p.fit_generator(generator=train_p_batch_generator,epochs = 3,verbose = 1) #sample_weight=pos_weight
    
    ############################################################
    
    discriminator = make_discriminator()
    generator = make_generator()
    opt_d = Adam(1e-4,beta_1 = 0.5,beta_2 = 0.9, decay= 0)
    opt_g = Adam(1e-4,beta_1 = 0.5,beta_2 = 0.9, decay= 0)
    
    ############ Make D() and G() for training ###################
    for layer in discriminator.layers:
        layer.trainable = False
    discriminator.trainable = False
    for layer in constrainer_e.layers:
        layer.trainable = False
    constrainer_e.trainable = False
    for layer in constrainer_p.layers:
        layer.trainable = False
    constrainer_p.trainable = False
    generator_input_noise = Input(shape=(100,))
    generator_input_label = Input(shape=(4,))
    generator_input = [generator_input_label,generator_input_noise]
    generator_layers = generator(generator_input)
    discriminator_input = generator_layers
    name_layer = tf.keras.layers.Lambda(lambda x: x, name='discriminator_output')
    discriminator_layers_for_generator = name_layer(discriminator(discriminator_input))

    # This is the constrainer output
    name_layer_2 = tf.keras.layers.Lambda(lambda x: x, name='constrainer_output')

    real_samples_constrainer = Input(shape=(74, 350, 1)) #shape=X_train.shape[1:])

    g_and_r = concatenate([real_samples_constrainer,generator_layers],axis = 0)

    g_and_r_out_p = constrainer_p(g_and_r)
    partial_regression_loss_p = partial(regression_loss,
                                       label = generator_input_label[:,:3],
                                       k = 1e-2* 10 * 16 * 2)
    g_and_r_out_e = constrainer_e(g_and_r)
    partial_regression_loss_e = partial(regression_loss,
                                     label=generator_input_label[:,3:4],
                                       k = 1e-4 *10 * 16 * 2 * 2)
    # Functions need names or Keras will throw an error
    partial_regression_loss_e.__name__ = 'regression_e'
    partial_regression_loss_p.__name__ = 'regression_p'

    g_and_r_out_e = tf.keras.layers.Lambda(lambda x: x, name='constrainer_e')(g_and_r_out_e)
    g_and_r_out_p = tf.keras.layers.Lambda(lambda x: x, name='constrainer_p')(g_and_r_out_p)
                                                
    generator_model = Model(inputs=[generator_input_label,
                                    generator_input_noise,
                                    real_samples_constrainer],
                            outputs=[discriminator_layers_for_generator,
                                     g_and_r_out_e,
                                     g_and_r_out_p])
    #generator_model.compile(optimizer = opt_g, loss = {'discriminator_output':wasserstein_loss})
    generator_model.compile(optimizer = opt_g, loss = [wasserstein_loss, 
                                                       partial_regression_loss_e,
                                                       partial_regression_loss_p])
    generator_model.metrics_names
    ################################ Discriminator model ############################
    for layer in discriminator.layers:
        layer.trainable = True
    for layer in generator.layers:
        layer.trainable = False
    discriminator.trainable = True
    generator.trainable = False
    real_samples = Input(shape=(74, 350, 1))
    critic_input_label = Input(shape=(4,))
    print('real_samples: ',real_samples)
    generator_input_noise_for_discriminator = Input(shape=(100,))
    generator_input_for_discriminator = [critic_input_label,generator_input_noise_for_discriminator]
    generated_samples_for_discriminator = generator(generator_input_for_discriminator)
    print('generated_samples_for_discriminator: ',generated_samples_for_discriminator)
    discriminator_output_from_generator = tf.keras.layers.Lambda(lambda x: x, name='gen_pred_output')(discriminator(generated_samples_for_discriminator))
    print('discriminator_output_from_generator: ',generated_samples_for_discriminator.shape)
    discriminator_output_from_real_samples = tf.keras.layers.Lambda(lambda x: x, name='real_samples_output')(discriminator(real_samples))

    averaged_samples = RandomWeightedAverage()([real_samples,
                                                generated_samples_for_discriminator])

    averaged_samples_out = tf.keras.layers.Lambda(lambda x: x, name='averaged_output')(discriminator(averaged_samples))
    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)

    #constrainer_output_from_generator = tf.keras.layers.Lambda(lambda x: x, name='constrainer_output')(constrainer_e(generated_samples_for_discriminator))
    # Functions need names or Keras will throw an error
    partial_gp_loss.__name__ = 'gradient_penalty'

    discriminator_model = Model(inputs=[real_samples,
                                        critic_input_label,
                                        generator_input_noise_for_discriminator],
                                outputs=[discriminator_output_from_real_samples,
                                         discriminator_output_from_generator,
                                         averaged_samples_out])
    discriminator_model.compile(optimizer=opt_d, loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss])
    discriminator_model.metrics_names


    for layer in constrainer_e.layers:
        layer.trainable = True
    constrainer_e.trainable = True
    for layer in constrainer_p.layers:
        layer.trainable = True
    constrainer_p.trainable = True

    critic = []
    dis = []
    con_p,con_e = [],[]
    eval_critic = []
    ################################### Training #####################################
    start = 0
    iterations = train_e_batch_generator.__len__()
    indices = np.array([i for i in range(iterations - 1)])
    for epoch in range(start,start+Epochs):

        print("Epoch: ", epoch)
        print("Number of batches: ", iterations)
        discriminator_loss = []
        generator_loss = []
        constrainer_loss_e = []
        constrainer_loss_p = []
        minibatches_size = BATCH_SIZE * TRAINING_RATIO
        np.random.shuffle(indices)
        print(indices)
        for i in range(iterations // TRAINING_RATIO):
            if i % 100 == 0: print('Minibatch %i processed.' %i,)
            minibatches = indices[i:i+5]
            for j in range(TRAINING_RATIO):
                image_batch,label_batch = train_batch_generator.__getitem__(minibatches[j])
                batch_size = image_batch.shape[0]
                noise = np.random.rand(batch_size, 100).astype(np.float32)
                positive_y = np.ones((batch_size, 1), dtype=np.float32)
                negative_y = -positive_y
                dummy_y = np.zeros((batch_size, 1), dtype=np.float32)
                discriminator_loss.append(discriminator_model.train_on_batch(
                    [image_batch,label_batch, noise],
                    [positive_y, negative_y, dummy_y]))
            #Maybe training constrainer with generator
            constrainer_loss_e.append(constrainer_e.train_on_batch([image_batch],
                                                                         [label_batch[:,3]])) 
            constrainer_loss_p.append(constrainer_p.train_on_batch([image_batch],
                                                                         [label_batch[:,:3]])) #,sample_weight=weight_batch
            generator_loss.append(generator_model.train_on_batch([label_batch,
                                                                  np.random.rand(batch_size,100),
                                                                  image_batch],
                                                                [positive_y,dummy_y,dummy_y]))
        critic += discriminator_loss
        dis += generator_loss
        con_e += constrainer_loss_e
        con_p += constrainer_loss_p
        save_loss(critic,dis,con_e,con_p,"gen-wTh"+str(iterations))
        save_models(save_model_path,"gen-wTh"+str(iterations),generator,discriminator,constrainer_e,constrainer_p)
        save_weights(save_model_path,"gen-wTh"+str(iterations),generator_model,discriminator_model,constrainer_e,constrainer_p)
    
if __name__ == "__main__":
    train(100)


