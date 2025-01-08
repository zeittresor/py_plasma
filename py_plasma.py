import pygame
import numpy as np
import sys
import math
from numba import njit, prange

pygame.init()
infoObject = pygame.display.Info()
screen_width, screen_height = infoObject.current_w, infoObject.current_h
screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
pygame.display.set_caption("plasma - github.com/zeittresor")
clock = pygame.time.Clock()
FPS = 60
scale_factor = 1
scaled_width = screen_width // scale_factor
scaled_height = screen_height // scale_factor

x = np.linspace(0, 4 * np.pi, scaled_width, dtype=np.float32)
y = np.linspace(0, 4 * np.pi, scaled_height, dtype=np.float32)
X, Y = np.meshgrid(x, y)

initial_frequencies = np.array([1.0, 1.5, 2.0, 2.5], dtype=np.float32)
num_waves = len(initial_frequencies)
phases = np.zeros(num_waves, dtype=np.float32)
phase_speeds = np.array([0.02, 0.015, 0.01, 0.025], dtype=np.float32)
amp_changes = np.array([0.001, 0.0015, 0.002, 0.0012], dtype=np.float32)
freq_changes = np.array([0.0005, 0.0007, 0.0006, 0.0004], dtype=np.float32)
amplitudes = initial_frequencies.copy()
frequencies = initial_frequencies.copy()
wind_time = 0.0
wind_direction_frequency = 0.1
wind_strength_frequency = 0.05
hue_shift = 0.0
hue_shift_speed = 0.0005
interf_frequencies = np.array([0.5, 1.0, 1.5], dtype=np.float32)
interf_phases = np.zeros(len(interf_frequencies), dtype=np.float32)
interf_phase_speeds = np.array([0.01, 0.012, 0.015], dtype=np.float32)
interf_amp_changes = np.array([0.0005, 0.0006, 0.0007], dtype=np.float32)
interf_freq_changes = np.array([0.0002, 0.0003, 0.0004], dtype=np.float32)
interf_amplitudes = interf_frequencies.copy()
interf_frequencies_current = interf_frequencies.copy()

@njit(parallel=True)
def generate_plasma_numba(X, Y, frequencies, phases, amplitudes, wind_strength, wind_direction):
    height, width = X.shape
    plasma = np.zeros((height, width), dtype=np.float32)
    
    wind_x = wind_strength * math.cos(wind_direction)
    wind_y = wind_strength * math.sin(wind_direction)
    
    for i in prange(len(frequencies)):
        freq = frequencies[i]
        phase = phases[i]
        amp = amplitudes[i]
        for y_idx in range(height):
            for x_idx in range(width):
                shifted_x = X[y_idx, x_idx] + wind_x
                shifted_y = Y[y_idx, x_idx] + wind_y
                if i < len(frequencies) - 1:
                    plasma[y_idx, x_idx] += amp * math.sin(freq * shifted_x + phase) + amp * math.sin(freq * shifted_y + phase)
                else:
                    plasma[y_idx, x_idx] += amp * math.sin(freq * shifted_y + phase)
    
    min_val = plasma.min()
    max_val = plasma.max()
    if max_val - min_val != 0:
        plasma = (plasma - min_val) / (max_val - min_val)
    else:
        plasma = np.zeros_like(plasma)
    
    return plasma

@njit(parallel=True)
def generate_interference_mask_numba(X, Y, frequencies, phases, amplitudes):
    height, width = X.shape
    mask = np.zeros((height, width), dtype=np.float32)
    
    for i in prange(len(frequencies)):
        freq = frequencies[i]
        phase = phases[i]
        amp = amplitudes[i]
        for y_idx in range(height):
            for x_idx in range(width):
                mask[y_idx, x_idx] += amp * math.sin(freq * X[y_idx, x_idx] + phase) + amp * math.sin(freq * Y[y_idx, x_idx] + phase)
    
    min_val = mask.min()
    max_val = mask.max()
    if max_val - min_val != 0:
        mask = (mask - min_val) / (max_val - min_val)
    else:
        mask = np.zeros_like(mask)
    
    return mask

@njit(parallel=True)
def plasma_to_rgb_numba(plasma, hue_shift):
    height, width = plasma.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in prange(height):
        for x in prange(width):
            hue = (plasma[y, x] + hue_shift) % 1.0
            saturation = 1.0
            value = 1.0

            h = hue * 6.0
            i = int(math.floor(h)) % 6
            f = h - math.floor(h)
            p = 0.0
            q = 1.0 - f
            t = f

            if i == 0:
                r, g, b = value, t, p
            elif i == 1:
                r, g, b = q, value, p
            elif i == 2:
                r, g, b = p, value, t
            elif i == 3:
                r, g, b = p, q, value
            elif i == 4:
                r, g, b = t, p, value
            elif i == 5:
                r, g, b = value, p, q

            rgb[y, x, 0] = int(r * 255)
            rgb[y, x, 1] = int(g * 255)
            rgb[y, x, 2] = int(b * 255)
    
    return rgb

@njit(parallel=True)
def apply_brightness_mask_numba(rgb, mask):
    height, width, _ = rgb.shape
    for y in prange(height):
        for x in prange(width):
            brightness = mask[y, x] * 0.5 + 0.5
            rgb[y, x, 0] = min(int(rgb[y, x, 0] * brightness), 255)
            rgb[y, x, 1] = min(int(rgb[y, x, 1] * brightness), 255)
            rgb[y, x, 2] = min(int(rgb[y, x, 2] * brightness), 255)
    return rgb

plasma = generate_plasma_numba(X, Y, frequencies, phases, amplitudes, 0.0, 0.0)
rgb_plasma = plasma_to_rgb_numba(plasma, hue_shift)

interference_mask = generate_interference_mask_numba(X, Y, interf_frequencies, interf_phases, interf_amplitudes)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
                break

    wind_time += 1.0 / FPS

    wind_direction = math.sin(wind_time * wind_direction_frequency) * math.pi  # Bereich: [-π, π]
    wind_strength = 0.05 + 0.05 * math.sin(wind_time * wind_strength_frequency)  # Bereich: [0, 0.1]

    for idx in range(len(phases)):
        phases[idx] += phase_speeds[idx]
        amplitudes[idx] += amp_changes[idx]
        frequencies[idx] += freq_changes[idx]
        if amplitudes[idx] > initial_frequencies[idx] * 2:
            amp_changes[idx] = -amp_changes[idx]
        elif amplitudes[idx] < initial_frequencies[idx] / 2:
            amp_changes[idx] = -amp_changes[idx]
        
        if frequencies[idx] > initial_frequencies[idx] * 2:
            freq_changes[idx] = -freq_changes[idx]
        elif frequencies[idx] < initial_frequencies[idx] / 2:
            freq_changes[idx] = -freq_changes[idx]

    for idx in range(len(interf_phases)):
        interf_phases[idx] += interf_phase_speeds[idx]
        interf_amplitudes[idx] += interf_amp_changes[idx]
        interf_frequencies_current[idx] += interf_freq_changes[idx]
        if interf_amplitudes[idx] > interf_frequencies[idx] * 2:
            interf_amp_changes[idx] = -interf_amp_changes[idx]
        elif interf_amplitudes[idx] < interf_frequencies[idx] / 2:
            interf_amp_changes[idx] = -interf_amp_changes[idx]
        
        if interf_frequencies_current[idx] > interf_frequencies[idx] * 2:
            interf_freq_changes[idx] = -interf_freq_changes[idx]
        elif interf_frequencies_current[idx] < interf_frequencies[idx] / 2:
            interf_freq_changes[idx] = -interf_freq_changes[idx]

    plasma = generate_plasma_numba(X, Y, frequencies, phases, amplitudes, wind_strength, wind_direction)
    hue_shift = (hue_shift + hue_shift_speed) % 1.0
    rgb_plasma = plasma_to_rgb_numba(plasma, hue_shift)
    interference_mask = generate_interference_mask_numba(X, Y, interf_frequencies, interf_phases, interf_amplitudes)
    rgb_plasma = apply_brightness_mask_numba(rgb_plasma, interference_mask)
    plasma_surface = pygame.surfarray.make_surface(rgb_plasma.swapaxes(0, 1))
   
    if scale_factor != 1:
        plasma_surface = pygame.transform.scale(plasma_surface, (screen_width, screen_height))
    
    screen.blit(plasma_surface, (0, 0))
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()
