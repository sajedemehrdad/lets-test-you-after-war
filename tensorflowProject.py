import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# ۱. داده‌ها رو لود کن + نرمالایز
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0

# ۲. مدل خیلی ساده (Sequential = مثل scikit-learn خطی)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),     # ۲۸×۲۸ → ۷۸۴
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')    # ۱۰ کلاس خروجی
])

# ۳. کامپایل (مثل fit در scikit)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ۴. آموزش ( epochs=۱۰ کافیه برای ~۹۸٪)
model.fit(x_train, y_train, epochs=10, validation_split=0.1)

# ۵. تست
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"دقت روی تست: {test_acc:.4f}")

# ۶. پیش‌بینی یک عکس تصادفی (برای دیدن)
rnd = np.random.randint(0, len(x_test))
pred = model.predict(x_test[rnd:rnd+1])
print("پیش‌بینی:", np.argmax(pred))
plt.imshow(x_test[rnd], cmap='gray')
plt.show()