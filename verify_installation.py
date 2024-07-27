import tensorflow as tf
from PIL import Image
import roboflow

try:
    import flask
    flask_version = flask.__version__
except ImportError:
    flask_version = "Flask is not installed"

print("TensorFlow version:", tf.__version__)
print("Pillow version:", Image.__version__)
print("Roboflow is installed.")
print("Flask version:", flask_version)
