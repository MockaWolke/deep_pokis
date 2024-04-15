import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np

def draw_bbox(image_path, bboxes, fig_size = 4):
    """
    Draws bounding boxes on the image specified.

    :param image_path: Path to the image file.
    :param bboxes: List of bounding boxes in YOLO format [(center_x, center_y, width, height), ...].
    """
    # Open an image file
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        img_width, img_height = img.size

        for bbox in bboxes:
            if len(bbox) == 4:
            
                center_x, center_y, width, height = bbox
            else:
                _, center_x, center_y, width, height = bbox
                
                
            # Convert from YOLO to PIL format
            left = (center_x - width / 2) * img_width
            top = (center_y - height / 2) * img_height
            right = (center_x + width / 2) * img_width
            bottom = (center_y + height / 2) * img_height

            # Draw rectangle on image
            draw.rectangle([left, top, right, bottom], outline="red", width=2)

        # Display the image
        plt.figure(figsize=(fig_size,fig_size))
        plt.imshow(np.array(img))
        plt.axis("off")
        plt.show()
