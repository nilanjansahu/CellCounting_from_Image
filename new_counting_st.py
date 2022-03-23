import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import exposure
from skimage.util import img_as_ubyte
from skimage import io
from skimage import (color, feature, filters, measure, segmentation)
import streamlit as st
import pandas as pd
from streamlit_cropper import st_cropper
from PIL import Image

st.set_page_config(
        page_title="Count Cells",
        page_icon="🖖",
        
    )
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
img_file = st.sidebar.file_uploader("Choose a file")
realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
aspect_dict = {
    "1:1": (1, 1),
    "16:9": (16, 9),
    "4:3": (4, 3),
    "2:3": (2, 3),
    "Free": None
}
aspect_ratio = aspect_dict[aspect_choice]

if img_file:
    img = Image.open(img_file).convert('L')
    if not realtime_update:
        st.write("Double click to save crop")
    # Get a cropped image from the frontend
    box = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                aspect_ratio=aspect_ratio, return_type='box')
    
    image = io.imread(img_file, as_gray = True)
    image = image[box['top']:box['top'] + box['height'], box['left']:box['left'] + box['width']]
    


image = img_as_ubyte(image)
clip_limit_ = st.slider('clip_limit for equalize adaptive hist', -1.0, 1.0, 0.005, 0.001) # scienticfic name the exact method
image = exposure.equalize_adapthist(image, clip_limit = clip_limit_)
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
ax.set_title('Image')
ax.axis('off')
st.pyplot(fig)
class_ = st.slider('multi-otsu classes', 1, 5, 3, 1)
thresholds = filters.threshold_multiotsu(image, classes=class_)
regions = np.digitize(image, bins=thresholds)

fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].hist(image.ravel(), bins=255)
ax[0].set_title('Histogram of pixel values')
for thresh in thresholds:
    ax[0].axvline(thresh, color='y')
ax[0].axis('off')
ax[1].imshow(regions)
ax[1].set_title('Multi-Otsu thresholding')
ax[1].axis('off')
#st.header('Mean: '+str(np.mean(image.ravel())))
#st.header('Standard Deviation: '+str(np.std(image.ravel())))
st.pyplot(fig)

threshold = st.slider('which class to take', 1, class_, 1, 1)
cells = image > thresholds[threshold-1]
dividing = image > thresholds[1]
labeled_cells = measure.label(cells)
labeled_dividing = measure.label(dividing)
naive_mi = labeled_dividing.max() / labeled_cells.max()
#print(naive_mi,thresholds[0],thresholds[1])

fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
ax[0].imshow(image)
ax[0].set_title('Original')
ax[0].axis('off')
ax[2].imshow(cells)
ax[2].set_title('All cells')
ax[2].axis('off')
ax[1].imshow(dividing)
ax[1].set_title('Joint cells')
ax[1].axis('off')
st.pyplot(fig)



distance = ndi.distance_transform_cdt(cells)

fig, ax = plt.subplots()
ax.imshow(distance, cmap='gray')
ax.set_title('Microscopy image')
ax.axis('off')
st.pyplot(fig)

min_distance_ = st.slider('clip_limit', 0, 50, 20, 1)
local_max_coords = feature.peak_local_max(distance, min_distance = min_distance_)
#print(len(local_max_coords))
local_max_mask = np.zeros(distance.shape, dtype=bool)
local_max_mask[tuple(local_max_coords.T)] = True
markers = measure.label(local_max_mask)

segmented_cells = segmentation.watershed(-distance, markers, mask=cells)
#print(segmented_cells)
fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].imshow(cells, cmap='gray')
ax[0].set_title('Overlapping cells')
ax[0].axis('off')
ax[1].imshow(color.label2rgb(segmented_cells, bg_label=0))
ax[1].set_title('Segmented cells')
ax[1].axis('off')
st.pyplot(fig)
table = measure.regionprops_table(segmented_cells,properties=['label', 'area', 'perimeter', 'axis_major_length', 'axis_minor_length'])
st.write('Total number of cells ' + str(segmented_cells.max()))
st.dataframe(pd.DataFrame(table))
#print(len(measure.find_contours(segmented_cells)))
