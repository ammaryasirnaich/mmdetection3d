from transformers import ViTImageProcessor, ViTModel, ViTFeatureExtractor
from PIL import Image
import requests
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torchvision
import numpy as np



# Load pre-trained ViT model and feature extractor
# model = ViTModel.from_pretrained('google/vit-base-patch16-224',output_attentions=True, add_pooling_layer=False, attn_implementation="eager")
# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

model = ViTModel.from_pretrained('facebook/dino-vits8',output_attentions=True, add_pooling_layer=False, attn_implementation="eager")
feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits8')

# Preprocess an example image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

plt.imshow(image)
# image
# model

'''
inputs = feature_extractor(images=image, return_tensors="pt")
### Forward pass
with torch.no_grad():
    outputs = model(**inputs)

#### Extract attention maps from the last layer
attentions = outputs.attentions[-1]  # Shape: (batch_size, num_heads, seq_len, seq_len)
print(attentions.shape)
##### Visualize attention maps for each head


attentions = outputs.attentions[-1] # we are only interested in the attention maps of the last layer
nh = attentions.shape[1] # number of head
#### patch size 28 x28 = 784 for image (224x224)
#### we keep only the output patch attention
attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
print(attentions.shape)




threshold = 0.6
pixel_values = inputs.pixel_values 
w_featmap = pixel_values.shape[-2] // model.config.patch_size
h_featmap = pixel_values.shape[-1] // model.config.patch_size

#### we keep only a certain percentage of the mass
val, idx = torch.sort(attentions)
val /= torch.sum(val, dim=1, keepdim=True)
cumval = torch.cumsum(val, dim=1)
th_attn = cumval > (1 - threshold)
idx2 = torch.argsort(idx)
for head in range(nh):
    th_attn[head] = th_attn[head][idx2[head]]
th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
#### interpolate
th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=model.config.patch_size, mode="nearest")[0].cpu().numpy()

attentions = attentions.reshape(nh, w_featmap, h_featmap)
attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=model.config.patch_size, mode="nearest")[0].cpu()
attentions = attentions.detach().numpy()




#### show and save attentions heatmaps
output_dir = '.'
os.makedirs(output_dir, exist_ok=True)
torchvision.utils.save_image(torchvision.utils.make_grid(pixel_values, normalize=True, scale_each=True), os.path.join(output_dir, "img.png"))
for j in range(nh):
    fname = os.path.join(output_dir, "attn-head" + str(j) + ".png")
    plt.figure()
    plt.imshow(attentions[j])
    plt.imsave(fname=fname, arr=attentions[j], format='png')
    #print(f"{fname} saved.")
    
    
'''