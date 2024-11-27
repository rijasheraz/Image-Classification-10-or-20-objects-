import pygame
import numpy as np
import cv2 as cv
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from tkinter import Tk, messagebox, filedialog

# Initialize Pygame for the loading screen
pygame.init()
screen = pygame.display.set_mode((700, 500))
pygame.display.set_caption("Image Classification")

# Colors for loading page
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

# Loading page function
def loading_page():
    screen.fill(WHITE)

    # Title Text
    font = pygame.font.SysFont("timesnewroman", 30, bold=True)
    title_text = font.render(" Image Classification", True, BLACK)
    screen.blit(title_text, (screen.get_width() // 2 - title_text.get_width() // 2, 20))

    # Loading Image
    loading_image = pygame.image.load("D:\\ai\\image recognizer\\image Recognization.png")
    loading_image = pygame.transform.scale(loading_image, (400, 300))
    screen.blit(loading_image, (screen.get_width() // 2 - 200, screen.get_height() // 2 - 150))

    # Progress bar
    bar_width = 650
    bar_height = 25
    progress_bar_rect = pygame.Rect(
        (screen.get_width() // 2 - bar_width // 2), screen.get_height() - bar_height - 20, bar_width, bar_height
    )
    pygame.draw.rect(screen, BLACK, progress_bar_rect, 2)

    total_steps = 100
    for i in range(total_steps):
        pygame.draw.rect(
            screen, BLUE, (progress_bar_rect.x, progress_bar_rect.y, (i * bar_width) // total_steps, bar_height)
        )
        pygame.display.flip()
        pygame.time.delay(30)

    pygame.time.delay(1000)

# Display the loading page
loading_page()
pygame.quit()

# Caltech-101 class names (first 30 classes)
class_names = [
    'airplanes', 'cars', 'faces', 'motorbikes', 'watchs', 'leopards', 'horses', 'elephants', 'dolphins', 'sharks',
    'palm_trees', 'sunglasses', 'starfish', 'butterflies', 'roosters', 'cannon', 'umbrella', 'windsurfing', 'bicycles',
    'flamingos', 'skyscrapers', 'chairs', 'cougar_body', 'cougar_face', 'crab', 'grasshoppers', 'scissors', 'keyboard', 'panda'
]

# Load the Caltech-101 dataset from a directory
def load_caltech_101_data(data_dir, num_classes=30):
    images = []
    labels = []

    # Load images from each class folder
    for label, class_name in enumerate(class_names[:num_classes]):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):  # Ensure the directory exists
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.jpg') or img_name.endswith('.jpeg'):
                    img_path = os.path.join(class_dir, img_name)
                    img = cv.imread(img_path)
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    img_resized = cv.resize(img, (32, 32))  # Resize to 32x32
                    images.append(img_resized)
                    labels.append(label)

    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Normalize images to [0, 1]
    images = images / 255.0
    return images, labels

# Set the data directory for Caltech-101 images
data_dir = "D:\\ai\\image recognizer\\model\\caltech-101"  # Full directory of the Caltech-101 dataset

# Load Caltech-101 dataset
images, labels = load_caltech_101_data(data_dir)

# Convert labels to one-hot encoding
labels = to_categorical(labels, len(class_names))

# Split the data into training and testing sets
training_images, testing_images, training_labels, testing_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Build a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')  # Adjust for 30 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

# Evaluate the model
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Save the model
model.save('caltech_image_classifier.keras')

# Load the trained model
model = models.load_model('caltech_image_classifier.keras')

# Function to preprocess and predict an image
def predict_image(image_path):
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_resized = cv.resize(img, (32, 32))

    prediction = model.predict(np.array([img_resized]) / 255.0)
    confidence_threshold = 0.8
    index = np.argmax(prediction)
    confidence = prediction[0][index]

    if confidence >= confidence_threshold:
        result_text = f'Prediction: {class_names[index]} with confidence {confidence:.2f}'
    else:
        result_text = f'Prediction: Other class with confidence {confidence:.2f}'

    print(result_text)

    plt.imshow(img)
    plt.title(result_text)
    plt.show()

# Main loop for GUI-based interaction
def main():
    root = Tk()
    root.withdraw()  # Hide the main window

    # Ask the user to select an initial image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        messagebox.showinfo("No File Selected", "Exiting the program. Goodbye!")
        return

    # Predict the initial image
    predict_image(file_path)

    # Loop to ask for additional predictions
    while True:
        answer = messagebox.askyesno("Predict Another Image", "Do you want to predict another image?")
        if answer:
            # Open file dialog to select another image
            file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
            if file_path:
                predict_image(file_path)
            else:
                messagebox.showinfo("No File Selected", "No image selected. Exiting.")
                break
        else:
            # Exit the program
            messagebox.showinfo("Goodbye", "Exiting the program. Goodbye!")
            break

# Run the main loop
if __name__ == "__main__":
    main()
