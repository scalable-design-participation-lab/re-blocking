#!/usr/bin/env python
"""
Ground Truth Generation for the Building-to-Parcel Workflow
Leonard Schrage, l.schrage@northeastern.edu / lschrage@mit.edu, March 2025

This script generates ground truth images for parcels and buildings over a set of samples 
(e.g., in Brooklyn) using the buffered parcel area as the image extent.
Each parcel is assigned a unique random color, and a parcel-to-color mapping is saved 
(as JSON) for use in the Voronoi script. Two sets of outputs are created:
  - A set of parcel images showing only parcels that contain at least one building.
  - A set of building images (if requested) showing building footprints, where each building 
    is split by parcel boundaries so that each fragment is colored with the corresponding 
    parcel's color.
If no parcels with buildings are found within a sample's buffered area, that sample is skipped.
The same sample indices are used for both processes.

Usage:
    python ground_truth_generation.py --buildings <BUILDINGS_PATH> --parcels <PARCELS_PATH> \
        --output-dir <OUTPUT_DIR> --sample-file <SAMPLE_FILE> --num-samples <NUM_SAMPLES> \
        [--with-buildings] [--color-mapping <COLOR_MAPPING_FILE>]

Parameters:
  - BUILDINGS_PATH: Path to the buildings shapefile.
  - PARCELS_PATH: Path to the parcels shapefile.
  - OUTPUT_DIR: Base output directory.
  - SAMPLE_FILE: Path to the sample indices file.
  - NUM_SAMPLES: Number of samples to process.
  - --with-buildings (optional): Generate building images.
  - --color-mapping (optional): File path to save the parcel-to-color mapping 
       (default: parcel_color_mapping.json).
"""

import os
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import warnings
import cartopy.crs as ccrs
from cartopy.io.img_tiles import MapboxTiles
from dotenv import load_dotenv
from pathlib import Path
import json
import gc

# Load environment variables and suppress warnings
load_dotenv()
warnings.filterwarnings('ignore')

#########################################################
# CONFIGURATION
#########################################################

# Input file paths (update these as needed)
BUILDINGS_PATH = "/home/ls/sites/re-blocking/data/shapefiles/ny-manhattan-buildings/geo_export_a80ea1a2-e8e0-4ffd-862c-1199433ac303.shp"
PARCELS_PATH = "/home/ls/sites/re-blocking/data/shapefiles/ny-manhattan-parcels/NYC_2021_Tax_Parcels_SHP_2203/Kings_2021_Tax_Parcels_SHP_2203.shp"

# Output directories and files
OUTPUT_DIR = "brooklyn_comparison"
PARCELS_DIR = os.path.join(OUTPUT_DIR, "parcels")
BUILDINGS_DIR = os.path.join(OUTPUT_DIR, "buildings")
SAMPLE_FILE = os.path.join(OUTPUT_DIR, "brooklyn_samples.npy")
DEFAULT_COLOR_MAPPING_FILE = os.path.join(OUTPUT_DIR, "parcel_color_mapping.json")

# Processing parameters
BUFFER_DISTANCE = 200      # Buffer (in map units) around each parcel
NUM_SAMPLES = 1000         # Total number of samples to process

# Visualization settings
FIGURE_SIZE = (7, 7)
DPI = 96
SATELLITE_ZOOM = 18
EXTENT_SCALE_FACTOR = 0.7

#########################################################
# UTILITY FUNCTIONS
#########################################################

def random_hex_color(seed=None):
    """Generate a random hex color, optionally with a seed for reproducibility."""
    if seed is not None:
        random.seed(int(seed))
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def render_map(geometries, colors, area_geometry, output_path, buildings_gdf=None, include_buildings=False):
    """
    Render a map using Mapbox satellite imagery with the given geometries and colors.
    Uses the same extent calculation as the Voronoi script.
    Optionally overlays building footprints.
    """
    mapbox_token = os.environ.get('MAPBOX_ACCESS_TOKEN')
    if not mapbox_token:
        print("Warning: MAPBOX_ACCESS_TOKEN not found in environment variables")
        return

    fig = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    tiler = MapboxTiles(mapbox_token, 'satellite-v9')
    ax = fig.add_subplot(1, 1, 1, projection=tiler.crs)
    
    bounds = area_geometry.bounds
    centroid_x = (bounds[2] + bounds[0]) / 2
    centroid_y = (bounds[3] + bounds[1]) / 2
    dist1 = bounds[2] - bounds[0]
    dist2 = bounds[3] - bounds[1]
    max_dist = max(dist1, dist2) / 2
    scaled_max_dist = max_dist * EXTENT_SCALE_FACTOR
    ax.set_extent([
        centroid_x - scaled_max_dist,
        centroid_x + scaled_max_dist,
        centroid_y - scaled_max_dist,
        centroid_y + scaled_max_dist
    ], crs=ccrs.epsg('3857'))
    
    ax.add_image(tiler, SATELLITE_ZOOM)
    
    for geom, color in zip(geometries, colors):
        ax.add_geometries([geom], crs=ccrs.epsg('3857'), facecolor=color, edgecolor='none', alpha=1.0)
    
    if include_buildings and buildings_gdf is not None:
        for _, row in buildings_gdf.iterrows():
            ax.add_geometries([row.geometry], crs=ccrs.epsg('3857'),
                              facecolor='none', edgecolor='black', linewidth=0.5, alpha=0.7)
    
    ax.set_axis_off()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=DPI)
    plt.close(fig)
    gc.collect()

def save_color_mapping(parcels_df, mapping_file):
    """
    Save the parcel-to-color mapping as a JSON file.
    The mapping keys are the parcel indices (as strings) and values are the colors.
    """
    mapping = {str(idx): color for idx, color in zip(parcels_df.index, parcels_df['color'])}
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved parcel color mapping to {mapping_file}")

def process_ground_truth_sample(i, parcel_idx, parcels_df, buildings_df, output_dirs):
    """
    Process one sample:
      - Create a buffered area around the chosen parcel.
      - Use that buffered area as the image extent.
      - Subset parcels and buildings that intersect the buffered area.
      - Perform a spatial join to filter parcels so that only those containing buildings are kept.
      - If no parcels with buildings are found, skip the sample.
      - Render a parcels image.
      - Render a buildings image where each building is split by parcel boundaries so each fragment gets the correct color.
    """
    try:
        parcel = parcels_df.loc[parcel_idx]
        area_geometry = parcel.geometry.buffer(BUFFER_DISTANCE * 0.75)
        
        # Subset all parcels and buildings that intersect the buffered area
        all_parcels = parcels_df[parcels_df.geometry.intersects(area_geometry)]
        all_buildings = buildings_df[buildings_df.geometry.intersects(area_geometry)]
        
        # Use a spatial join to filter parcels that actually contain buildings
        if not all_buildings.empty:
            joined = gpd.sjoin(all_parcels, all_buildings[['geometry']], how="inner", predicate="intersects")
            parcels_with_buildings = all_parcels.loc[joined.index.unique()]
        else:
            parcels_with_buildings = gpd.GeoDataFrame(columns=all_parcels.columns)
        
        if parcels_with_buildings.empty:
            print(f"Sample {i}: No parcels with buildings found within the area, skipping.")
            return
        
        # Render parcel image using only parcels with buildings
        parcel_output = os.path.join(output_dirs['parcels'], f"parcels_{i:06d}.jpg")
        render_map(parcels_with_buildings.geometry.tolist(), parcels_with_buildings['color'].tolist(), area_geometry, parcel_output)
        
        # Render building image with splitting
        if output_dirs.get('buildings'):
            building_fragments = []
            fragment_colors = []
            for idx, building_row in all_buildings.iterrows():
                # Split each building by intersecting it with the parcels that contain buildings.
                fragments = []
                for p_idx, p_row in parcels_with_buildings.iterrows():
                    inter = building_row.geometry.intersection(p_row.geometry)
                    if not inter.is_empty:
                        if inter.geom_type == 'GeometryCollection':
                            polys = [geom for geom in inter if geom.geom_type in ['Polygon', 'MultiPolygon']]
                            for poly in polys:
                                fragments.append((poly, p_row.color))
                        else:
                            fragments.append((inter, p_row.color))
                for frag, col in fragments:
                    building_fragments.append(frag)
                    fragment_colors.append(col)
            
            if not building_fragments:
                print(f"Sample {i}: No building fragments produced, skipping building image.")
            else:
                building_output = os.path.join(output_dirs['buildings'], f"buildings_{i:06d}.jpg")
                render_map(building_fragments, fragment_colors, area_geometry, building_output)
        
        print(f"Sample {i} processed successfully.")
    except Exception as e:
        print(f"Error processing sample {i} (parcel index {parcel_idx}): {e}")

def generate_ground_truth(buildings_path, parcels_path, output_dir, sample_file, num_samples,
                          generate_buildings, color_mapping_file):
    """
    Main function to generate ground truth images and the parcel-to-color mapping.
    It uses the buffered parcel area as the image extent and creates:
      - A set of parcel images showing only parcels that contain buildings.
      - A set of building images (if --with-buildings is specified) where each building is split
        so that each fragment is colored with the corresponding parcel's color.
    It also generates (or loads) sample indices.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(PARCELS_DIR, exist_ok=True)
    if generate_buildings:
        os.makedirs(BUILDINGS_DIR, exist_ok=True)
    
    print(f"Loading building data from: {buildings_path}")
    if not os.path.exists(buildings_path):
        print("Building shapefile not found.")
        return
    buildings_df = gpd.read_file(buildings_path).to_crs(3857)
    print(f"Loaded {len(buildings_df)} buildings.")
    
    print(f"Loading parcel data from: {parcels_path}")
    if not os.path.exists(parcels_path):
        print("Parcel shapefile not found.")
        return
    parcels_df = gpd.read_file(parcels_path).to_crs(3857)
    print(f"Loaded {len(parcels_df)} parcels.")
    
    if "color" not in parcels_df.columns:
        parcels_df["color"] = [random_hex_color(i) for i in range(len(parcels_df))]
    
    save_color_mapping(parcels_df, color_mapping_file)
    
    if os.path.exists(sample_file):
        sampled_indices = np.load(sample_file)
        print(f"Loaded {len(sampled_indices)} sample indices from {sample_file}")
    else:
        if len(parcels_df) >= num_samples:
            sampled_indices = np.random.choice(parcels_df.index, num_samples, replace=False)
        else:
            sampled_indices = parcels_df.index.tolist()
            print(f"Warning: Only {len(parcels_df)} parcels available, using all.")
        np.save(sample_file, sampled_indices)
        print(f"Saved {len(sampled_indices)} sample indices to {sample_file}")
    
    total_samples = len(sampled_indices)
    output_dirs = {'parcels': PARCELS_DIR}
    if generate_buildings:
        output_dirs['buildings'] = BUILDINGS_DIR
    
    for i, parcel_idx in enumerate(tqdm(sampled_indices, desc="Processing samples")):
        process_ground_truth_sample(i, parcel_idx, parcels_df, buildings_df, output_dirs)
    
    print("Completed generating ground truth images.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Ground Truth Images for the Building-to-Parcel Workflow with Consistent Parcel Colors")
    parser.add_argument('--buildings', type=str, default=BUILDINGS_PATH, help='Path to buildings shapefile')
    parser.add_argument('--parcels', type=str, default=PARCELS_PATH, help='Path to parcels shapefile')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR, help='Base output directory')
    parser.add_argument('--sample-file', type=str, default=SAMPLE_FILE, help='Path to sample indices file')
    parser.add_argument('--num-samples', type=int, default=NUM_SAMPLES, help='Number of samples to process')
    parser.add_argument('--with-buildings', action='store_true', help='Generate building images')
    parser.add_argument('--color-mapping', type=str, default=DEFAULT_COLOR_MAPPING_FILE, help='File path to save parcel-to-color mapping')
    
    args = parser.parse_args()
    generate_ground_truth(
        buildings_path=args.buildings,
        parcels_path=args.parcels,
        output_dir=args.output_dir,
        sample_file=args.sample_file,
        num_samples=args.num_samples,
        generate_buildings=args.with_buildings,
        color_mapping_file=args.color_mapping
    )