from PIL import Image
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
    image_paths = [f'{i+1}_map.png' for i in range(4)]
    create_gif(image_paths, 'output.gif', duration=500)
