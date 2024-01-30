# import torch
# from torchvision import transforms
# from PIL import Image
# from models.base_block import FeatClassifier, BaseClassifier
# from models.resnet import resnet50
# from config import argument_parser
# from tools.utils import load_ckpt, load

# # Load the model architecture
# backbone = resnet50()
# num_classes = 26  # Replace "your_num_of_classes" with the number of classes in your dataset
# classifier = BaseClassifier(nattr=num_classes)
# model = FeatClassifier(backbone, classifier)

# # Load the trained model weights
# model_path = '/content/PAR/Strong_Baseline_of_Pedestrian_Attribute_Recognition/checkpoints/ckpt_max.pth'  # Specify the path to your saved model
# model.load_state_dict(torch.load(model_path))
# print(model_state_dict.keys())
# model.eval()  # Set the model to evaluation mode

# # Load and preprocess the single image
# image_path = 'test/1.jpg'  # Specify the path to your single image
# image = Image.open(image_path)
# transform = transforms.Compose([
#     transforms.Resize((256, 128)),  # Resize to match the expected input size of the model
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Assuming ImageNet normalization
# ])

# image = transform(image)
# image = image.unsqueeze(0)  # Add batch dimension

# # Perform inference
# with torch.no_grad():
#     output = model(image.cuda() if torch.cuda.is_available() else image)

# # Process the output
# probabilities = torch.softmax(output, dim=1)[0]
# predicted_class = torch.argmax(probabilities).item()

# # You can map the predicted class index to your class labels
# # For example, if you have a list of class labels:
# class_labels = ['bag', 'hat', 'upper_1', 'upper_2', 'upper_3', 'upper_4', 'upper_5',
#        'upper_6', 'upper_7', 'upper_8', 'upper_9', 'upper_10', 'upper_11',
#        'lower_1', 'lower_2', 'lower_3', 'lower_4', 'lower_5', 'lower_6',
#        'lower_7', 'lower_8', 'lower_9', 'lower_10', 'lower_11', 'gender_0',
#        'gender_1']  # Replace with your actual class labels
# predicted_label = class_labels[predicted_class]

# print(f'Predicted class: {predicted_label}, Probability: {probabilities[predicted_class].item()}')


import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from dataset.AttrDataset import AttrDataset, get_transform
from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50
from config import argument_parser
from tools.function import get_pedestrian_metrics
from tools.utils import load_ckpt
import numpy as np

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 128)),  # Resize to match the expected input size of the model
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Assuming ImageNet normalization
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def validate_single_image(model, image_path):
    model.eval()  # Set the model to evaluation mode
    
    # Preprocess the image
    image = preprocess_image(image_path)
    image = image.cuda() if torch.cuda.is_available() else image
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
    
    # Process the output
    probabilities = torch.sigmoid(output)[0].cpu().numpy()
    
    return probabilities

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    
    # Load the model architecture
    backbone = resnet50()
    classifier = BaseClassifier(nattr=26)
    model = FeatClassifier(backbone, classifier)
    
    # Load the checkpoint
    ckpt_file = 'checkpoints/ckpt_max.pth'  # Specify the path to your saved checkpoint file
    load_ckpt([model], ckpt_file)
    
    # Path to the single image for validation
    image_path = 'test/1.jpg'  # Specify the path to your single image
    
    # Validate the single image
    probabilities = validate_single_image(model, image_path)
    
    # Print or use the probabilities as needed
    print("Predicted Probabilities:", probabilities)