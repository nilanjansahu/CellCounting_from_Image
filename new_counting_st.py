import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import exposure
from skimage.util import img_as_ubyte
from skimage import io
from skimage import color, feature, filters, measure, segmentation, morphology
import streamlit as st
import pandas as pd
from streamlit_cropper import st_cropper
from PIL import Image
from streamlit_drawable_canvas import st_canvas


st.set_page_config(
        page_title="Count Cells",
        page_icon="🖖",
        layout="wide"
    )
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 1])

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
image = []
if img_file:
    img = Image.open(img_file)  #.convert('L')
    if not realtime_update:
        st.write("Double click to save crop")
    # Get a cropped image from the frontend
    with col1:
        cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                aspect_ratio=aspect_ratio)
    bg_image = cropped_img
    
  
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")


realtime_update = st.sidebar.checkbox("Update in realtime", True)



# Create a canvas component
to_merge =[]
with col1:
    canvas_result = st_canvas(
        fill_color = "rgba(255, 165, 0, 0)",  # Fixed fill color with some opacity
        stroke_width = stroke_width,
        stroke_color = stroke_color,
        background_color = bg_color,
        background_image = bg_image,
        update_streamlit = realtime_update,
        height = bg_image.size[1],
        width = bg_image.size[0],
        drawing_mode = drawing_mode,
        point_display_radius = point_display_radius if drawing_mode == 'point' else 0,
        key="canvas",
    )
    to_merge = canvas_result.image_data


#st.image()


r, g, b, a = Image.fromarray(np.uint8(to_merge)).split()

transformed_rgb = Image.merge("RGB", (r, g, b))
transformed_mask = Image.merge("L", (a,))
bg_image.paste(transformed_rgb, transformed_mask)
bg_image = bg_image.convert('L')

image = img_as_ubyte(bg_image)
with col2:
    with st.expander("adaptive histogram"):
        clip_limit_ = st.slider('clip_limit for equalize adaptive hist', -1.0, 1.0, 0.005, 0.001) 
        image = exposure.equalize_adapthist(image, clip_limit = clip_limit_)
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        ax.set_title('Image')
        ax.axis('off')
        st.pyplot(fig)
with col2:
    expander = st.expander("otsu thresholding")
class_ = expander.slider('multi-otsu classes', 1, 5, 3, 1)
thresholds = filters.threshold_multiotsu(image, classes=class_)
regions = np.digitize(image, bins=thresholds)

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(regions)
ax.set_title('Multi-Otsu thresholding')
ax.axis('off')
#st.header('Mean: '+str(np.mean(image.ravel())))
#st.header('Standard Deviation: '+str(np.std(image.ravel())))
expander.pyplot(fig)

threshold = expander.slider('which class to take', 1, class_, 1, 1)
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
expander.pyplot(fig)



distance = ndi.distance_transform_cdt(cells)

fig, ax = plt.subplots()
ax.imshow(distance, cmap='gray')
ax.set_title('Microscopy image')
ax.axis('off')
expander.pyplot(fig)

min_distance_ = expander.slider('clip_limit', 0, 50, 20, 1)
local_max_coords = feature.peak_local_max(distance, min_distance = min_distance_)
#print(len(local_max_coords))
local_max_mask = np.zeros(distance.shape, dtype=bool)
local_max_mask[tuple(local_max_coords.T)] = True
markers = measure.label(local_max_mask)

segmented_cells = segmentation.watershed(-distance, markers, mask=cells)


table = measure.regionprops_table(segmented_cells,properties=['label', 'area', 'perimeter', 'centroid', 'axis_major_length', 'axis_minor_length'])
min_area = st.slider('area minimum limit', min_value=1, max_value=int(max(table['area'])), value=50, step=1)
segmented_cells = morphology.remove_small_objects(segmented_cells, min_area)

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(color.label2rgb(segmented_cells, bg_label=0))
ax.set_title('Segmented cells')
ax.axis('off')
table = measure.regionprops_table(segmented_cells,properties=['label', 'area', 'perimeter', 'centroid', 'axis_major_length', 'axis_minor_length'])
for i in range(len(table['label'])):
    ax.text(table['centroid-1'][i], table['centroid-0'][i], table['label'][i], horizontalalignment='center',
                verticalalignment='center', fontsize=7, color='w')
with col2:
    st.pyplot(fig)
    st.write('Total number of cells ' + str(segmented_cells.max()))
with col2:
    st.pyplot(fig)
    st.write('Total number of cells ' + str(segmented_cells.max()))
with col2:
    st.pyplot(fig)
    st.write('Total number of cells ' + str(segmented_cells.max()))

st.dataframe(pd.DataFrame(table))

# probability error
fig, ax = plt.subplots(figsize=(10, 10))
ax.hist(image.ravel(), bins=255)
ax.set_title('Histogram of pixel values')
for thresh in thresholds:
    ax.axvline(thresh, color='y')
ax.axis('off')
st.pyplot(fig)
