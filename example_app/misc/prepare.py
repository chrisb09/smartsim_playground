#!/usr/bin/env python3


def plot_data(data, width, filename):
    import numpy as np
    # Data is a 1D numpy array, but it represents a 2D grid where each entry has 2 values: (ground_height, water_height), and we have image_width * image_height entries.
    height = data.shape[0] // (width * 2)  # Divide by 2 because data is interleaved
    print("Width * Height = " + (width * height).__str__())
    print("Data[0::2] size: " + data[0::2].shape.__str__())
    ground_heights = data[0::2].reshape((height, width))
    water_heights = data[1::2].reshape((height, width))
    
    # We want to create an rgb image where we first draw the ground heights colored like a terrain map, and then overlay the water heights we omit for now
    
    # we create gradual color transitions for the ground heights
    elevation_ground = 144
    end_of_green = 154
    end_of_brown = 185
    # after that, we go to white at 256
    
    color_map_ground = np.zeros((256, 3), dtype=np.uint8)
    
    # Vectorized color map creation
    idx = np.arange(256)
    
    # used to highlight sea level
    #mask = idx == elevation_ground - 1
    #color_map_ground[mask] = np.column_stack([255, 0, 0])
    
    # Grayscale for low elevations (water/sea)
    mask = idx < elevation_ground
    color_map_ground[mask] = np.column_stack([idx[mask], idx[mask], idx[mask]])
    
    # Green shades
    mask = (idx >= elevation_ground) & (idx < end_of_green)
    green_value = (128 + 127 * (idx[mask] - elevation_ground) / (end_of_green - elevation_ground)).astype(np.uint8)
    color_map_ground[mask] = np.column_stack([np.zeros_like(green_value), green_value, np.zeros_like(green_value)])
    
    # Brown shades
    mask = (idx >= end_of_green) & (idx < end_of_brown)
    brown_value = (255 * (idx[mask] - end_of_green) / (end_of_brown - end_of_green)).astype(np.uint8)
    color_map_ground[mask] = np.column_stack([brown_value, np.full_like(brown_value, 128), np.zeros_like(brown_value)])
    
    # White shades
    mask = idx >= end_of_brown
    white_value = (200 + 55 * (idx[mask] - end_of_brown) / (256 - end_of_brown)).astype(np.uint8)
    color_map_ground[mask] = np.column_stack([white_value, white_value, white_value])
    
    from PIL import Image
    
    print("Applying color map...")
    # Vectorized pixel lookup - clamp ground_heights to [0, 255]
    gh_clamped = np.clip(ground_heights, 0, 255).astype(np.uint8)
    
    # Use color map as lookup table for entire image at once
    rgb_image = color_map_ground[gh_clamped]
    
    
    # We use the water data to color based on water heights (blueish)
    # Very simple for now: if water height > 0, we add a blue tint proportional to water height
    print("Applying water overlay...")
    wh_clamped = np.clip(water_heights, 0, 255).astype(np.uint8)
    blue_tint = (wh_clamped * 0.5).astype(np.uint8)  # scale down the blue tint
    
    rgb_image[:, :, 2] = np.clip(rgb_image[:, :, 2] + blue_tint, 0, 255)
    
    
    
    print("Creating image with dimensions:", rgb_image.shape)
    img = Image.fromarray(rgb_image, 'RGB')
    print("Saving image...")
    # Deactivate for now, thumbnail is sufficient for testing
    #img.save(filename)
    
    # Save a thumnail too with max size 
    img.thumbnail((4096, 4096))
    thumb_filename = filename.replace(".png", "_thumb.png")
    img.save(thumb_filename)
    
    
    
    print(f"Saved image to {filename}")
    
def generate_water_data(data):
    # We just put 10 water everywhere for now
    data[1::2] = 10  # water heights at odd indices
    
    
    
    
def save_data_to_file(data, filename):
    import h5py
    
    print(f"Saving data to HDF5 file {filename}...")
    
    with h5py.File(filename, "w") as f:
        f.create_dataset("heights", data=data)
    print(f"Saved data to {filename}")


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Prepare image data for SmartSim example application")
    
    parser.add_argument("image", type=str, help="Input image filename")
    parser.add_argument("output_image", type=str, help="Output image filename")
    parser.add_argument("hdf5_output", type=str, help="Output HDF5 filename")
    
    args = parser.parse_args()
    
    input_image_path = args.image
    
    if not os.path.exists(input_image_path):
        print(f"Input file {input_image_path} does not exist.")
        exit(1)
        
    import numpy as np
    
    from PIL import Image
    
    
    Image.MAX_IMAGE_PIXELS = None  # or set to a larger value like 500000000
    
    
    img = Image.open(input_image_path)
    data_raw = np.array(img)
    
    print(f"Loaded image of shape {data_raw.shape} from {input_image_path}")
    
    width = data_raw.shape[1]
    
    if len(data_raw.shape) == 3 and data_raw.shape[2] > 1:
        # Multi-channel image
        ground_data = data_raw[:, :, 0].flatten()  # Use only the first channel
    else:
        ground_data = data_raw.flatten()
    
    # Interleave ground heights with dummy water heights (0)
    data = np.zeros(ground_data.shape[0] * 2, dtype=ground_data.dtype)
    data[0::2] = ground_data  # ground heights at even indices
    data[1::2] = 0  # water heights at odd indices (dummy data)
    
    generate_water_data(data)
        
    print(f"Data flattened to shape {data.shape}")
    
    print("Numpy data type:", data.dtype)
    print("  " + data.nbytes.__str__() + " bytes")
    print("  "  +(data.nbytes / 1024).__str__() + " KiB")
    print("  "  +(data.nbytes / (1024*1024)).__str__() + " MiB")
    print("  "  +(data.nbytes / (1024*1024*1024)).__str__() + " GiB")
    print("  " + data[0].nbytes.__str__() + " bytes per entry")
    
    # We only need the first channel if it's multi-channel
    #if len(data_raw.shape) > 1:
    
    plot_data(data, width, filename=args.output_image)
    save_data_to_file(data, filename=args.hdf5_output)