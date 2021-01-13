import numpy as np


def padded_window(cells, x, y, radius, padding_value=0):
    """Slice a fixed size window from `cells`, centered on `(y, x)`. Borders are padded.

    Args:
        cells: Game state
        x, y: Position of the window
        radius: Radius of the window, resulting window has a shape of `(2*radius+1, 2*radius+1)`.
        padding_value: Scalar value to use for padding
    Return:
        window: ndarray with shape `(2*radius+1, 2*radius+1)`. May be a slice to `cells`.
    """
    height, width = cells.shape

    if radius <= x < width - radius and radius <= y < height - radius:
        # No padding neccessary, return direct slice
        window = cells[y - radius:y + radius + 1, x - radius:x + radius + 1]
    else:  # Padding required
        # Initialize window with padding and copy the visible cells
        window = np.empty((radius * 2 + 1, radius * 2 + 1), dtype=cells.dtype)
        window[:] = padding_value
        window[max(0, radius - y):min(radius * 2 + 1, height + radius - y),
               max(0, radius - x):min(radius * 2 + 1, width + radius - x)] = \
            cells[max(0, y - radius):min(height, y + radius + 1),
                  max(0, x - radius):min(width, x + radius + 1)]
    return window
