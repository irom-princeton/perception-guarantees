from PIL import Image
import numpy as np
def create_gif(image_paths, output_gif_path, duration=500):
    """Creates a GIF from a list of image paths."""

    images = [Image.open(image_path) for image_path in image_paths]
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=duration,
        loop=0  # 0 means infinite loop
    )

if __name__ == '__main__':
    # image_paths = [f'{i+1}_map.png' for i in range(17)]
    image_paths = [f'coverage_{i:.2f}.png' for i in np.arange(0.25,0.99,0.01)]
    create_gif(image_paths, 'output.gif', duration=500)
