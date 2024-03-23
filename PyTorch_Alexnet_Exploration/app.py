import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from model_file import AlexNet  

def preprocessing_image(image): # preprocessing of image
    preprocess = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    image = preprocess(image).unsqueeze(0) # batch dimension
    return image

model = AlexNet(num_classes=10)
model.load_state_dict(torch.load('PyTorch_Alexnet_Exploration/alexnet_model.pth', map_location=torch.device('cpu')))
st.title('Image Classification')
file_upload = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"]) # uploader for image

if file_upload is not None:
    image = Image.open(file_upload) # display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    input_tensor = preprocessing_image(image)
    with torch.no_grad():
        model.eval()  
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        prediction = predicted.item()

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    st.write(f'Prediction: {class_names[prediction]}')
