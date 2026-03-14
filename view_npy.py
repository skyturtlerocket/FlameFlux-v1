#!/usr/bin/env python3.10

import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2

def view_npy_file(file_path, title=None, cmap='gray'):
    """View a NPY file using matplotlib"""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return
    
    #read the NPY file
    img = np.load(file_path)
    
    if img is None:
        print(f"Error: Could not read {file_path}")
        return
    
    print(f"Image shape: {img.shape}")
    print(f"Data type: {img.dtype}")
    print(f"Min value: {img.min()}")
    print(f"Max value: {img.max()}")
    print(f"Mean value: {img.mean():.4f}")
    print(f"Standard deviation: {img.std():.4f}")
    
    #handle NaN values
    if np.any(np.isnan(img)):
        print(f"Warning: {np.sum(np.isnan(img))} NaN values found")
        img_display = img.copy()
        img_display[np.isnan(img_display)] = img_display[~np.isnan(img_display)].min()
    else:
        img_display = img
    
    #display the image
    plt.figure(figsize=(12, 8))
    
    #for multi-channel images, show first 3 channels as RGB
    if len(img.shape) == 3 and img.shape[2] > 1:
        if img.shape[2] >= 3:
            #show first 3 channels as RGB
            rgb_img = img_display[:, :, :3]
            plt.imshow(rgb_img)
            plt.title(f"{title or os.path.basename(file_path)} (RGB from first 3 channels)")
        else:
            #single channel
            plt.imshow(img_display[:, :, 0], cmap=cmap)
            plt.title(f"{title or os.path.basename(file_path)} (Channel 0)")
    else:
        #single channel image
        plt.imshow(img_display, cmap=cmap)
        plt.title(title or os.path.basename(file_path))
    
    plt.colorbar(label='Pixel Value')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def list_available_npy():
    """List all available NPY files in the data directory"""
    print("Available NPY files:")
    print("\nTerrain data (in each fire directory):")
    terrain_files = ['dem.npy', 'aspect.npy', 'slope.npy', 'ndvi.npy', 
                     'band_2.npy', 'band_3.npy', 'band_4.npy', 'band_5.npy']
    for file in terrain_files:
        print(f"  {file}")
    
    print("\nFire perimeter data (in perims/ subdirectory):")
    print("  Date-based files (e.g., 0711.npy, 0712.npy, etc.)")
    
    #list fires
    fires = [f for f in os.listdir('data') if os.path.isdir(os.path.join('data', f)) and not f.startswith('.')]
    print(f"\nAvailable fires: {fires}")
    
    #show some example files
    print("\nExample files found:")
    for fire in fires[:3]:  # Show first 3 fires
        fire_dir = f"data/{fire}"
        if os.path.exists(fire_dir):
            terrain_files = [f for f in os.listdir(fire_dir) if f.endswith('.npy') and not f.startswith('.')]
            if terrain_files:
                print(f"  {fire}: {terrain_files[:3]}...")  # Show first 3 files
            
            perims_dir = f"data/{fire}/perims"
            if os.path.exists(perims_dir):
                perim_files = [f for f in os.listdir(perims_dir) if f.endswith('.npy') and not f.startswith('.')]
                if perim_files:
                    print(f"    perims: {perim_files[:5]}...")  # Show first 5 files
    
    #check for madre perims
    print("\nMadre perims (converted):")
    madre_perims_dir = "data/madre/perims"
    if os.path.exists(madre_perims_dir):
        madre_files = [f for f in os.listdir(madre_perims_dir) if f.endswith('.npy') and not f.startswith('.')]
        if madre_files:
            print(f"  Available: {madre_files}")
        else:
            print("  No NPY files found")
    else:
        print("  Directory not found")

def get_colormap_for_data_type(data_type):
    """Return appropriate colormap based on data type"""
    colormaps = {
        'dem': 'terrain',      # Elevation data
        'slope': 'plasma',     # Slope data
        'aspect': 'hsv',       # Aspect data (circular)
        'ndvi': 'RdYlGn',      # Vegetation index
        'band_2': 'gray',      # Landsat bands
        'band_3': 'gray',
        'band_4': 'gray',
        'band_5': 'gray',
        'perim': 'Reds'        # Fire perimeter
    }
    return colormaps.get(data_type, 'gray')

def main():
    parser = argparse.ArgumentParser(description='View NPY files in the FireCast project')
    parser.add_argument('--file', type=str, help='Path to NPY file to view')
    parser.add_argument('--fire', type=str, help='Fire name (e.g., beaverCreek)')
    parser.add_argument('--type', type=str, choices=['dem', 'aspect', 'slope', 'ndvi', 'band_2', 'band_3', 'band_4', 'band_5'], 
                       help='Type of terrain data to view')
    parser.add_argument('--date', type=str, help='Date for perimeter data (e.g., 0711)')
    parser.add_argument('--madre', type=str, help='Date for madre perimeter data (e.g., 0707)')
    parser.add_argument('--list', action='store_true', help='List available NPY files')
    parser.add_argument('--cmap', type=str, default=None, help='Colormap to use (e.g., gray, terrain, plasma)')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_npy()
        return
    
    if args.file:
        cmap = args.cmap or get_colormap_for_data_type('default')
        view_npy_file(args.file, cmap=cmap)
    elif args.fire and args.type:
        file_path = f"data/{args.fire}/{args.type}.npy"
        cmap = args.cmap or get_colormap_for_data_type(args.type)
        view_npy_file(file_path, f"{args.fire} - {args.type}", cmap=cmap)
    elif args.fire and args.date:
        file_path = f"data/{args.fire}/perims/{args.date}.npy"
        cmap = args.cmap or get_colormap_for_data_type('perim')
        view_npy_file(file_path, f"{args.fire} - Perimeter {args.date}", cmap=cmap)
    elif args.madre:
        file_path = f"data/madre/perims/{args.madre}.npy"
        cmap = args.cmap or get_colormap_for_data_type('perim')
        view_npy_file(file_path, f"Madre - Perimeter {args.madre}", cmap=cmap)
    else:
        print("Usage examples:")
        print("  python3.10 view_npy.py --list")
        print("  python3.10 view_npy.py --fire beaverCreek --type dem")
        print("  python3.10 view_npy.py --fire beaverCreek --type ndvi --cmap RdYlGn")
        print("  python3.10 view_npy.py --fire beaverCreek --date 0711")
        print("  python3.10 view_npy.py --madre 0707")
        print("  python3.10 view_npy.py --file data/beaverCreek/dem.npy")
        print("\nAvailable colormaps: gray, terrain, plasma, hsv, RdYlGn, Reds, Blues, etc.")

if __name__ == "__main__":
    main() 