from keras.models import Model
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Multiply
from keras.layers.convolutional import Conv2D
from keras.optimizers import RMSprop
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.losses import huber_loss

def build_deep_q_model(
    image_size: tuple=(84,84),
    num_frames: int=4,
    num_actions: int=6,
    loss=huber_loss,
    optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
) -> Model:
    """
    Args:
        image_size: the shape of the image
        num_frames: the number of frames being stacked together
        num_actions: the output shape for the model, represents
                     the number of discrete actions available to a game
        loss: loss metric
        optimizer: optimizer for reducing error from batches

    Returns:
        blank DeepMind CNN for image classification in a reinforcement Agent
    """
    cnn_input = Input((*image_size, num_frames), name='cnn')
    cnn = Lambda(lambda x: x / 255.0)(cnn_input)
    cnn = Conv2D(32, (8, 8), strides=(4,4))(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Conv2D(64, (4,4), strides=(2,2))(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Conv2D(64, (3,3), strides=(1,1))(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(512)(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Dense(num_actions)(cnn)
    mask_input = Input((num_actions,),name='mask')
    output = Multiply()([cnn, mask_input])
    model = Model(inputs=[cnn_input, mask_input], outputs=output)
    model.compile(loss=loss, optimizer=optimizer)

    return model

if __name__ == "__main__":
    model = build_deep_q_model()
    print(model)
    print(123)
