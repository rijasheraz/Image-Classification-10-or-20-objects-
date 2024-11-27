import pygame
import numpy as np
import cv2 as cv
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tkinter import Tk  # Added this import
from tkinter.filedialog import askopenfilename
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

# CIFAR-10 class names
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Load and preprocess the data
(training_images, training_labels), (testing_images, testing_labels) = cifar10.load_data()

# Normalize image data to the range 0-1
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# Convert labels to one-hot encoding
training_labels = to_categorical(training_labels, 10)
testing_labels = to_categorical(testing_labels, 10)

'''
# Build a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
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
model.save('image_classifier.keras')
'''

# Load the trained model
model = models.load_model('image_classifier.keras')

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
    #plt.axis('off')
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
    