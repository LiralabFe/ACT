import math
import time


class LowPassFilter:
    def __init__(self, cutoff_frequency, initial_value=0.0):
        self.cutoff_frequency = cutoff_frequency
        self.y = initial_value
        self.last_time = time.perf_counter()

    def update(self, x):
        # Calcola il dt dall'ultima chiamata
        now = time.perf_counter()
        dt = now - self.last_time
        self.last_time = now

        # Filtro passa-basso del primo ordine
        rc = 1.0 / (2.0 * math.pi * self.cutoff_frequency)
        alpha = dt / (rc + dt)

        self.y += alpha * (x - self.y)

        return self.y
