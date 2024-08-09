import io
import random

import gaussian_mixture
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

random.seed(0)


def get_gauss(
    flow: np.ndarray,
    mask: np.ndarray,
    radius: int = 10,
    sigma: int = 10,
    max_sample_attempts=30,
    seed=0,
):
    output_from, output_to = gaussian_mixture.generate(
        mask, flow, radius, sigma, max_sample_attempts, seed
    )

    random_colors = np.random.rand(len(output_from), 3)

    img_curr = generate_gaussian_plot(
        output_from,
        mask.shape[0],
        mask.shape[1],
        random_colors,
        radius=10,
        save_png=False,
        save_path="cool.png",
        blur_radius=3.0,
    )

    img_next = generate_gaussian_plot(
        output_to,
        mask.shape[0],
        mask.shape[1],
        random_colors,
        radius=10,
        save_png=False,
        save_path="cool.png",
        blur_radius=3.0,
    )

    return img_curr, img_next


def generate_gaussian_plot(
    output: list[tuple[float, float]],
    height: int,
    width: int,
    colors,
    radius: float = 10.0,
    save_png=False,
    save_path="output.png",
    show=False,
    blur_radius=2.0,
):
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = plt.Axes(fig, (0.0, 0.0, 1.0, 1.0))
    fig.add_axes(ax)

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")

    ax.axis("off")

    xs, ys = zip(*output)
    
    ys_out = [height - y for y in ys]  # Flip y-coordinates

    ax.scatter(xs, ys_out, s=np.pi * radius**1.8, c=colors, marker="o", edgecolor="none")

    fig.canvas.draw()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)

    buf.seek(0)

    image_pil = Image.open(buf)
    image_pil = image_pil.convert("RGB")

    image_blurred = image_pil.filter(ImageFilter.GaussianBlur(blur_radius))

    if save_png:
        image_blurred.save(save_path, format="png")
    image_blurred = np.array(image_blurred)

    buf.close()

    if show:
        plt.imshow(image_blurred)
    else:
        plt.close(fig)
    return image_blurred
