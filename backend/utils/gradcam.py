import numpy as np
import tensorflow as tf
import cv2
import base64
from PIL import Image
import io

def generate_gradcam(model, img_array, layer_name=None):
    """
    Generate Grad-CAM heatmap for a given image array.
    img_array: preprocessed image array of shape (1, 224, 224, 3)
    Returns: base64 encoded heatmap image
    """
    # Find the last conv layer automatically if not specified
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break

    # Create gradient model
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv outputs
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize heatmap to 224x224
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Overlay on original image
    original = np.uint8(img_array[0] * 255)
    original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    superimposed = cv2.addWeighted(original_bgr, 0.6, heatmap_colored, 0.4, 0)

    # Convert to base64
    superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(superimposed_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return img_base64