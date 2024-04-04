#first attempt- give only reference and gamma passing rate as an input




def map_to_class(gamma_value):
    if gamma_value < 95:
        return 0
    elif 95 <= gamma_value < 96:
        return 1
    elif 96 <= gamma_value < 96.5:
        return 2
    elif 96.5 <= gamma_value < 97:
        return 3
    elif 97 <= gamma_value < 98:
        return 4
    elif 98 <= gamma_value < 99:
        return 5
    elif 99 <= gamma_value < 99.5:
        return 6
    elif 99.5 <= gamma_value < 99.8:
        return 7
    elif 99.8 <= gamma_value < 99.9:
        return 8
    elif 99.9 <= gamma_value <= 100:
        return 9
    else:
        return None
'''
df['class'] = df['gamma_txt'].apply(map_to_class)
print(df['class'])

df_train, df_test= train_test_split(df, test_size=0.2, random_state=42)

desired_size=(1024,1024)
X_train = load_and_preprocess_images(df_train['ref'], desired_size)
Y_train=df_train['class']
X_test = load_and_preprocess_images(df_test['ref'], desired_size)
Y_test=df_test['class']

train_Y_one_hot = to_categorical(Y_train)
test_Y_one_hot = to_categorical(Y_test)
train_X,valid_X,train_label,valid_label = train_test_split(X_train, train_Y_one_hot, test_size=0.2, random_state=13)

batch_size = 64
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
'''
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
'''