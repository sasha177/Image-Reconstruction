import numpy as np
from scipy.fftpack import fft, ifft, fftshift, fftfreq
from PIL import Image

def filter_sinogram(sinogram: np.ndarray) -> np.ndarray:
    proj_size, num_proj = sinogram.shape
    filtered_sino = np.zeros((proj_size, num_proj))

    for i in range(num_proj):
        proj = sinogram[:, i]
        proj_fft = fft(proj)
        freqs = fftfreq(len(proj_fft))

        B = np.max(np.abs(freqs))
        rfilt = np.abs(np.linspace(-B, B, proj_size))
        rfilt = fftshift(rfilt)

        proj_filtered = ifft(proj_fft * rfilt)
        filtered_sino[:, i] = np.real(proj_filtered)

    return filtered_sino

def backproject(sinogram, angles=None, save_gif=False):
    proj_size, num_proj = sinogram.shape
    reconstructed = np.zeros((proj_size, proj_size))
    frames = [] # for gif

    for i in range(num_proj):

        proj = sinogram[:, i]
        proj = np.expand_dims(proj, 0)

        dragged = np.repeat(proj, proj_size)
        dragged = dragged.reshape(proj_size, proj_size)
        dragged = dragged.T

        dragged_img = Image.fromarray(dragged)

        # assume one scan per degree, otherwise use angles
        theta = angles[i] if angles is not None else i
        dragged_img = dragged_img.rotate(theta, expand=False)
        reconstructed += np.array(dragged_img)

        # add partial reconstruction to frames
        if save_gif:
            frames.append(Image.fromarray(
                np.flipud(reconstructed) / np.max(reconstructed) * 255
            ).convert('RGB'))

    if save_gif:
        frames[0].save(
            'reconstruction.gif', format='GIF', 
            append_images = frames[1:], 
            save_all=True, duration=50, loop=0
        )

    # flip and normalize reconstructed image
    reconstructed = np.flipud(reconstructed)
    reconstructed /= np.max(reconstructed)

    return reconstructed

def fbp(sinogram, angles=None, save_gif=False):
    filtered_sino = filter_sinogram(sinogram)
    reconstructed = backproject(filtered_sino, angles, save_gif)
    return reconstructed

# calculate mean absolute error
def error(img, reconstructed):
    e_img = np.abs(img - reconstructed)
    return np.mean(e_img)
