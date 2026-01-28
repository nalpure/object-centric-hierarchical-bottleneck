import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image


def plot_images(images, save_path, labels=None, title=None, cmap=None):
    """
    Displays all images in a single row and saves the resulting plot.

    Args:
        images (iterable): An iterable of images. Each image should be of shape [3, H, W].
        save_path (str): File path to save the plotted image.
        labels (iterable): An iterable of labels for each image.
        title (str): Optional title for the entire figure.
        cmap (str): Optional colormap, e.g., 'gray' for grayscale images.
    """
    num_images = len(images)
    images = list(images)
    for i in range(num_images):
        if hasattr(images[i], 'detach'):
            images[i] = images[i].detach().cpu().numpy()
        
        images[i] = np.clip(images[i], 0, 1)

        if images[i].shape[0] == 1:
            images[i] = images[i].squeeze(0)  # shape: [H, W]

        if images[i].ndim == 3 and images[i].shape[0] == 3:
            images[i] = images[i].transpose(1, 2, 0)  # shape: [H, W, 3]
    
    # Create a figure with one row and as many columns as there are images.
    fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 4))
    
    # Ensure that axes is always iterable (if only one image, axes is not a list).
    if num_images == 1:
        axes = [axes]
    
    # Loop over images and display each one.
    for idx, img in enumerate(images):
        axes[idx].imshow(img, cmap=cmap)
        axes[idx].axis('off')
    
    if labels is not None:
        for ax, label in zip(axes, labels):
            ax.set_title(label)

    if title is not None:
        plt.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect parameter to add padding at the top
    plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_grid(rows, row_titles, column_titles, save_path="output.png"):
    num_columns = len(column_titles)
    
    if not num_columns == len(rows[0]):
        raise ValueError(f"Number of column titles ({num_columns}) must match number of columns in rows ({len(rows[0])}).")
    if not len(row_titles) == len(rows):
        raise ValueError(f"Number of row titles ({len(row_titles)}) must match number of rows ({len(rows)}).")
    
    fig, axes = plt.subplots(len(rows), num_columns, figsize=(4 * num_columns, 4 * len(rows)))

    if num_columns == 1:
        axes = axes.reshape(len(rows), 1)

    for row_idx, images in enumerate(rows):
        for col_idx in range(num_columns):
            img = images[col_idx]
            if hasattr(img, "detach"):
                img = img.detach().cpu().numpy()

            if img.shape[0] == 1:
                img = img.squeeze(0)  # shape: [H, W]
                cmap = "gray"
            elif img.ndim == 2:
                cmap = "gray"
            else:
                img = img.transpose(1, 2, 0)  # shape: [H, W, 3]
                cmap = None

            axes[row_idx, col_idx].imshow(img.clip(0, 1), cmap=cmap)
            axes[row_idx, col_idx].axis("off")
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(column_titles[col_idx], fontsize=12)

    # Manually add row titles on the left
    for row_idx, title in enumerate(row_titles):
        fig.text(0.1, 0.8 - row_idx * 0.76 / len(rows), title, va='center', ha='right', fontsize=14, weight='bold')

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def create_trail(img_seq, highlight="last"):
    """
    Creates a trailing effect by blending the highlighted image with
    a series of trailing images.

    Args:
        img_seq: numpy array of shape [T, H, W, C]
        highlight: 'last' or 'first' - which frame to highlight
    Returns:
        final_img: numpy array of shape [H, W, C]
    """
    if highlight == "last":
        highlighted_img = img_seq[-1]
    elif highlight == "first":
        highlighted_img = img_seq[0]
    else:
        raise ValueError(f"Unknown highlight mode: {highlight}")

    trail_length = img_seq.shape[1] - 1
    final_img = np.copy(highlighted_img) / 2.0

    for trail_img in img_seq[:-1]:
        final_img += (
            np.minimum(final_img, trail_img) / trail_length / 2.0
        )

    return final_img


def save_gif_from_array(frames, output_path="rollout.gif", fps=30, scale=4.0, loop=0):
    """
    Create a GIF directly from a list or NumPy array of RGB frames.
    
    Args:
        frames: list or np.ndarray of shape [T, H, W, 3] (values in [0,1] or [0,255])
        output_path: where to save the gif
        fps: frames per second
        scale: how much to enlarge frames (e.g. 2.0 = 2× bigger)
        loop: 0 = infinite, N = loop N times
    """
    # Convert to uint8 if needed
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
    
    if frames.ndim != 4:
        raise ValueError(f"Expected frames to have 4 dimensions [T, H, W, C], got {frames.shape}")
    
    if frames.shape[-1] > 10 and frames.shape[1] == 3:
        frames = frames.transpose(0, 2, 3, 1)  # Convert from [T, C, H, W] to [T, H, W, C]

    if scale != 1.0:
        new_frames = []
        for f in frames:
            img = Image.fromarray(f)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.Resampling.NEAREST)
            new_frames.append(np.array(img))
        frames = np.stack(new_frames)

    imageio.mimsave(output_path, frames, duration=1/fps, loop=loop)
    print(f"GIF saved to {output_path}")