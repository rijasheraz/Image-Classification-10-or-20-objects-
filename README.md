# Image-Classification-10-or-20-objects-
apply ai for train and test the model for later use

import pygame
import numpy as np
import cv2 as cv
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from tkinter import Tk, messagebox, filedialog
