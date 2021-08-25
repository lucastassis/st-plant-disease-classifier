import streamlit as st 
from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms as T


# Function for passing image through net
def run_img(model, img):
    transforms = T.Compose([T.Resize((224, 224)),
                                T.ToTensor(),
                                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = transforms(img)
    output = model(img.unsqueeze(0))
    return output

# Dictionary mapping class id to name and wikipedia page
name_dict = { 0 : ('Apple - Apple Scab', 'https://en.wikipedia.org/wiki/Apple_scab'),
              1 : ('Apple - Black Rot', 'https://en.wikipedia.org/wiki/Botryosphaeria_obtusa'),
              2 : ('Apple - Cedar Apple Rust', 'https://en.wikipedia.org/wiki/Gymnosporangium_juniperi-virginianae'),
              3 : ('Apple - Healthy',),
              4 : ('Grape - Black Rot', 'https://en.wikipedia.org/wiki/Black_rot_(grape_disease)'),
              5 : ('Grape - Esca (Black Measles)', 'https://en.wikipedia.org/wiki/Esca_(grape_disease)'),
              7 : ('Grape - Healthy',),
              6 : ('Grape - Leaf Blight (Isariopsis Leaf Spot)', 'https://en.wikipedia.org/wiki/Pseudocercospora_vitis')
            }

# Define net
model = torchvision.models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 8)
model.load_state_dict(torch.load('./weights/model_final.pth', map_location=torch.device('cpu')))
model.eval()

# Page title
st.set_page_config(page_title='Plant Disease Classifier', layout='centered')
st.markdown("# Plant Disease Classifier")

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type="jpg")

if uploaded_file is not None:
    # Upload and print image
    img = Image.open(uploaded_file)
    st.image(img)    
    # Run net
    if st.button('Make a prediction!'):
        output = run_img(model, img)
        output = torch.nn.functional.softmax(output, dim=1)
        label = int(output.argmax().numpy())
        confidence_level = round(output.detach().numpy()[0][label] * 100, 2)        
        
        # Show results
        st.success(f'***Disease:*** {name_dict[label][0]}\n\n ***Confidence level:*** {confidence_level}%')

        if label == 7 or label == 3:
            st.info('Your plant is healthy!')
        else:
            st.info(f'You can find more information about this disease here: [Link]({name_dict[label][1]})')
    




                