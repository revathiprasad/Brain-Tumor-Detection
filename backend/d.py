import tensorflow as tf

model = tf.saved_model.load("cnn_baseline_savedmodel")
infer = model.signatures["serving_default"]

print("Variables count:", len(infer.variables))
print("First variable mean:", infer.variables[0].numpy().mean())
