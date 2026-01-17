import tensorflow as tf
import traceback
import re
import warnings

MODEL_PATHS = {
    "resnet50": r"E:\\deep learning\\DEEP-LEARNING-2025-main\\DEEP-LEARNING-2025-main\\PROJECT_FILES\\notebooks\\resnet50_best.h5",
    "mobilenetv2": r"E:\\deep learning\\DEEP-LEARNING-2025-main\\DEEP-LEARNING-2025-main\\PROJECT_FILES\\notebooks\\mobilenetv2_best.h5",
    "custom_cnn": r"E:\\deep learning\\DEEP-LEARNING-2025-main\\DEEP-LEARNING-2025-main\\PROJECT_FILES\\notebooks\\custom_cnn_best.h5",
}


class PlaceholderLayer(tf.keras.layers.Layer):
    """A tolerant placeholder used only for model deserialization.

    It strips unknown config kwargs and implements an identity call.
    """
    def __init__(self, *args, **kwargs):
        # Keep only kwargs that Layer.__init__ expects
        allowed = {}
        for k in ("trainable", "dtype", "name"):
            if k in kwargs:
                allowed[k] = kwargs.pop(k)
        super().__init__(**allowed)

    def call(self, inputs, **kwargs):
        return inputs


def safe_load(path):
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        msg = str(e)
        names = re.findall(r"Unknown layer: '([^']+)'", msg)
        if not names:
            # try more general extraction of layer-like names
            names = re.findall(r"'([A-Za-z0-9_.]+Lambda)'", msg)
        if names:
            warnings.warn(f"Found unknown layers: {names}, trying placeholder Layer fallback")
            custom = {n: PlaceholderLayer for n in names}
            # common TF op names that commonly appear in h5 exports
            for common in ("TFOpLambda", "SlicingOpLambda", "tf.__operators__.getitem", "tf.math.truediv"):
                custom.setdefault(common, PlaceholderLayer)
            from tensorflow.keras.utils import custom_object_scope
            try:
                with custom_object_scope(custom):
                    return tf.keras.models.load_model(path, compile=False)
            except Exception:
                print("Fallback failed:")
                traceback.print_exc()
                raise
        else:
            print("Exception while loading:")
            traceback.print_exc()
            raise


if __name__ == '__main__':
    for name, path in MODEL_PATHS.items():
        print('\n--- Trying to load', name + ':', path)
        try:
            m = safe_load(path)
        except Exception as e:
            print(f"Failed to load {name}: {e}")
        else:
            print(f"Loaded {name} OK. Summary:")
            try:
                m.summary()
            except Exception:
                print("(Could not print model summary)")
