import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.layers import LeakyReLU
from sklearn.model_selection import train_test_split

#first nn- give only reference and gamma passing rate as an input; second nn- give reference and evaluation and gamma passing rate as an input


def model1(X_train, Y_train, X_test, Y_test):



    train_Y_one_hot = to_categorical(Y_train)
    test_Y_one_hot = to_categorical(Y_test)
    train_X,valid_X,train_label,valid_label = train_test_split(X_train, train_Y_one_hot, test_size=0.2, random_state=13)

    batch_size = 16
    epochs = 20
    num_classes = 10
    #print(X_train.shape)
    fashion_model = Sequential()
    fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(1024,1024,1),padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D((2, 2),padding='same'))
    fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))                  
    fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    fashion_model.add(Flatten())
    fashion_model.add(Dense(128, activation='linear'))
    fashion_model.add(LeakyReLU(alpha=0.1))                  
    fashion_model.add(Dense(num_classes, activation='softmax'))
    fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    fashion_model.summary()
    fashion_train = fashion_model.fit(X_train, train_Y_one_hot, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
    test_eval = fashion_model.evaluate(X_test, test_Y_one_hot, verbose=0)
    print('Test:', test_eval)
'''

#sieć- obrazek liczba
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=desired_size),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10)
test_loss, test_acc = model.evaluate(X_train,  Y_train, verbose=2)

print('\nTest accuracy:', test_acc)



fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
'''