# Animal Image Classification Web App
A Django web application for classifying animal images using a deep learning model based on ResNet152V2. The model is integrated into the backend and leverages image transformation techniques for preprocessing, ensuring high accuracy and robustness.

📌 Features
Upload and classify animal images through a user-friendly web interface.

Deep learning model powered by ResNet152V2, pre-trained on ImageNet and fine-tuned for animal species.

Image preprocessing using transformation techniques (resizing, normalization, etc.).

Clean and responsive frontend built with Django templates.

Real-time prediction with confidence scores.

🧠 Model Details
Architecture: ResNet152V2

Framework: TensorFlow / Keras

Training Dataset: Custom animal image dataset (or optionally, public datasets like CIFAR-100, iNaturalist, etc.)

Transformations Applied:

Image resizing

RGB normalization

Data augmentation (if applicable)

🛠️ Tech Stack

Component  |	Technology

Backend:	Django 4.x

Frontend:	HTML, CSS, Bootstrap

Model: Framework	TensorFlow/Keras

Image Handling:	Pillow, OpenCV
