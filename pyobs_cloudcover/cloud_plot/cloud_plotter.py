import io
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from cloudmap_rs import AltAzCoord


def plot_clouds(coordinates: List[AltAzCoord], clouds: List[bool]) -> bytes:
    '''
    z = 90 - np.rad2deg([coord.alt for coord in coordinates])
    az = np.rad2deg([coord.az for coord in coordinates])
    plt.polar()
    plt.scatter(az, z, c=clouds, s=1)
    plt.colorbar()
    '''

    lon, lat = [coord.az for coord in coordinates], [coord.alt for coord in coordinates]
    plt.polar()
    plt.scatter(lon, 90 - np.rad2deg(lat), c=clouds, s=1)
    plt.colorbar()

    return render_plot()

def render_plot() -> bytes:
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    return buffer.read()