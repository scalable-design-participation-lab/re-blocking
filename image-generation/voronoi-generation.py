#!/usr/bin/env python
"""
Voronoi Tessellation Generation for the Building-to-Parcel Workflow
Leonard Schrage, l.schrage@northeastern.edu / lschrage@mit.edu, March 2025
Based on: Fleischmann, Feliciotti, Romice, Porta (2019)

This script generates Voronoi tessellations over a set of samples (e.g., in Brooklyn)
by using building centroids as input points. It uses the same buffered parcel area (for
image extent) as the ground truth script. If block clipping is enabled, Voronoi cells are 
clipped to block boundaries (within the buffered area) without altering the image segment.
A pre-generated parcel-to-color mapping (saved as JSON) is used to assign colors to Voronoi 
cells. Additionally, if the --with-buildings flag is provided, the script generates a separate 
building overlay output. In this overlay, buildings that cross multiple parcels are split so 
that each fragment is colored with the color of the parcel it intersects.
If a buffered area does not contain any parcels with buildings, that sample is skipped.

Usage:
    python voronoi-generation.py --buildings <BUILDINGS_PATH> --parcels <PARCELS_PATH> \
        --blocks <BLOCKS_PATH> --output-dir <OUTPUT_DIR> --sample-file <SAMPLE_FILE> \
        --num-samples <NUM_SAMPLES> [--color-mapping <COLOR_MAPPING_FILE>] [--with-buildings]

Parameters:
  - BUILDINGS_PATH: Path to the buildings shapefile.
  - PARCELS_PATH: Path to the parcels shapefile.
  - BLOCKS_PATH: Path to the blocks shapefile.
  - OUTPUT_DIR: Base output directory.
  - SAMPLE_FILE: Path to the sample indices file.
  - NUM_SAMPLES: Number of samples to process.
  - --color-mapping: Path to the parcel-to-color mapping JSON file.
  - --with-buildings (optional): Generate additional output with building overlays.
"""

import os
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint
from shapely.ops import voronoi_diagram, unary_union
import random
import warnings
import cartopy.crs as ccrs
from cartopy.io.img_tiles import MapboxTiles
from dotenv import load_dotenv
import gc
from tqdm import tqdm
from pathlib import Path
import json
import sys  # Added for sys.exit

# Load environment variables and suppress warnings
load_dotenv()
warnings.filterwarnings('ignore')

#########################################################
# CONFIGURATION
#########################################################

# Input file paths (edit as needed)
BUILDINGS_PATH = "/home/ls/sites/re-blocking/data/shapefiles/ny-manhattan-buildings/geo_export_a80ea1a2-e8e0-4ffd-862c-1199433ac303.shp"
PARCELS_PATH = "/home/ls/sites/re-blocking/data/shapefiles/ny-manhattan-parcels/NYC_2021_Tax_Parcels_SHP_2203/Kings_2021_Tax_Parcels_SHP_2203.shp"
BLOCKS_PATH = "/home/ls/sites/re-blocking/data/shapefiles/ny-blocks/nyc_mappluto_24v4_1_unclipped_shp/MapPLUTO_UNCLIPPED.shp"

# Output directories and files
OUTPUT_DIR = "brooklyn_comparison"
VORONOI_DIR = os.path.join(OUTPUT_DIR, "voronoi")
VORONOI_BUILDINGS_DIR = os.path.join(OUTPUT_DIR, "voronoi_buildings")
SAMPLE_FILE = os.path.join(OUTPUT_DIR, "brooklyn_samples.npy")
DEFAULT_COLOR_MAPPING_FILE = os.path.join(OUTPUT_DIR, "parcel_color_mapping.json")

# Processing parameters
BUFFER_DISTANCE = 200      # Buffer (map units) around each parcel
NUM_SAMPLES = 1000         # Total number of samples to process
BATCH_SIZE = 100           # Process samples in batches
USE_BLOCKS = True          # Whether to use block boundaries for clipping Voronoi cells

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
    Uses the same extent calculation as the ground truth script.
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

def get_blocks(parcels_df, blocks_path):
    """
    Load blocks from the provided blocks_path and convert to the same CRS as parcels_df.
    If loading fails, return None.
    """
    if blocks_path and os.path.exists(blocks_path):
        try:
            blocks_df = gpd.read_file(blocks_path)
            blocks_df = blocks_df.to_crs(parcels_df.crs)
            print(f"Loaded {len(blocks_df)} blocks from {blocks_path}")
            return blocks_df
        except Exception as e:
            print(f"Error loading blocks: {e}")
    return None

def buffer_and_discretize_buildings(buildings_gdf, interval=1.0):
    """
    Use building centroids as input points for the Voronoi diagram.
    """
    points = []
    for idx, row in buildings_gdf.iterrows():
        centroid = row.geometry.centroid
        points.append((centroid, idx))
    return points

def simplified_voronoi(buildings_gdf, clip_geometry):
    """
    Create simplified Voronoi cells by buffering each building,
    then clip with the provided clip_geometry.
    """
    result_cells = []
    for idx, row in buildings_gdf.iterrows():
        try:
            buffer_dist = max(15, (row.geometry.area ** 0.5) * 0.5)
            cell = row.geometry.buffer(buffer_dist)
            cell = cell.intersection(clip_geometry)
            if cell.is_valid and not cell.is_empty and cell.area > 0:
                result_cells.append({
                    'building_id': idx,
                    'geometry': cell,
                    'color': random_hex_color(int(idx))  # Add a default color
                })
        except Exception as e:
            print(f"Error creating cell for building {idx}: {e}")
    return gpd.GeoDataFrame(result_cells, crs=buildings_gdf.crs)

def generate_voronoi_tessellation(buildings_gdf, clip_geometry):
    """
    Generate a Voronoi tessellation based on building centroids.
    Each cell is clipped with the provided clip_geometry.
    Falls back to the simplified method if an error occurs.
    """
    points_with_ids = buffer_and_discretize_buildings(buildings_gdf)
    if not points_with_ids:
        print("No valid points generated, falling back to simplified Voronoi.")
        return simplified_voronoi(buildings_gdf, clip_geometry)
    
    points = [pt for pt, _ in points_with_ids]
    point_ids = [bid for _, bid in points_with_ids]
    multipoint = MultiPoint(points)
    try:
        vor_polys = voronoi_diagram(multipoint, envelope=clip_geometry)
    except Exception as e:
        print(f"Error generating Voronoi diagram: {e}")
        return simplified_voronoi(buildings_gdf, clip_geometry)
    
    cells = []
    for i, poly in enumerate(vor_polys.geoms):
        building_id = point_ids[i] if i < len(point_ids) else None
        if building_id is not None:
            try:
                clipped = poly.intersection(clip_geometry)
                if clipped.is_valid and not clipped.is_empty:
                    # Initially assign a random color (will be overridden by mapping)
                    cells.append({
                        'building_id': building_id,
                        'geometry': clipped,
                        'color': random_hex_color(int(building_id))
                    })
            except Exception as e:
                print(f"Error processing cell for building {building_id}: {e}")
                continue
    if cells:
        return gpd.GeoDataFrame(cells, crs=buildings_gdf.crs)
    else:
        return simplified_voronoi(buildings_gdf, clip_geometry)

def split_building_by_parcels(building, parcels):
    """
    Splits a building geometry into fragments based on its intersection with each parcel.
    Returns a list of (fragment, color) tuples.
    """
    fragments = []
    for idx, row in parcels.iterrows():
        # Skip if parcel has no color
        if 'color' not in row:
            continue
            
        inter = building.intersection(row.geometry)
        if not inter.is_empty:
            if inter.geom_type == 'GeometryCollection':
                polys = [geom for geom in inter if geom.geom_type in ['Polygon', 'MultiPolygon']]
                for poly in polys:
                    fragments.append((poly, row.color))
            else:
                fragments.append((inter, row.color))
    return fragments

def process_sample(i, parcel_idx, parcels_df, buildings_df, blocks_df, output_dirs, generate_buildings, color_mapping):
    """
    Process one sample:
      - Create a buffered area around the chosen parcel.
      - Use that buffered area as the image extent.
      - If blocks are used, create a clip_geometry (the intersection of the buffered area and
        the union of blocks intersecting the area) for clipping Voronoi cells.
      - Subset parcels and buildings that intersect the buffered area.
      - Use a spatial join to filter parcels to those that contain at least one building.
      - If no parcels with buildings exist, skip the sample.
      - Generate Voronoi tessellation, then find which parcel each cell overlaps most with
        and assign the corresponding color from the color_mapping.
      - Render and save the Voronoi tessellation image.
      - If --with-buildings is specified, split each building by parcel boundaries so that each fragment gets 
        the correct color and render a separate building overlay image.
    """
    try:
        # Check if parcel_idx exists in parcels_df
        if parcel_idx not in parcels_df.index:
            print(f"Sample {i}: Parcel index {parcel_idx} not found in parcels dataframe, skipping.")
            return
            
        parcel = parcels_df.loc[parcel_idx]
        area_geometry = parcel.geometry.buffer(BUFFER_DISTANCE * 0.75)
        
        clip_geometry = area_geometry
        if USE_BLOCKS and blocks_df is not None and not blocks_df.empty:
            blocks_in_area = blocks_df[blocks_df.geometry.intersects(area_geometry)]
            if not blocks_in_area.empty:
                clip_geometry = area_geometry.intersection(unary_union(blocks_in_area.geometry.tolist()))
        
        all_parcels = parcels_df[parcels_df.geometry.intersects(area_geometry)]
        all_buildings = buildings_df[buildings_df.geometry.intersects(area_geometry)]
        
        # Filter to only those parcels that contain at least one building.
        if not all_buildings.empty:
            joined = gpd.sjoin(all_parcels, all_buildings[['geometry']], how="inner", predicate="intersects")
            parcels_with_buildings = all_parcels.loc[joined.index.unique()]
        else:
            parcels_with_buildings = gpd.GeoDataFrame(columns=all_parcels.columns)
        
        if parcels_with_buildings.empty:
            print(f"Sample {i}: No parcels with buildings found in the area, skipping.")
            return
        
        # Similarly, if no buildings at all, skip the sample.
        if all_buildings.empty:
            print(f"Sample {i}: No buildings found in the area, skipping.")
            return
        
        # Ensure parcels have color values from the color mapping
        parcels_with_buildings['color'] = parcels_with_buildings.index.map(
            lambda idx: color_mapping.get(str(idx), random_hex_color(int(idx)))
        )
        
        # Generate Voronoi tessellation from all buildings.
        voronoi_cells = generate_voronoi_tessellation(all_buildings, clip_geometry)
        if voronoi_cells.empty:
            print(f"Sample {i}: No Voronoi cells generated, skipping.")
            return
        
        # For each Voronoi cell, find the parcel with maximum overlap and use its color
        cell_colors = {}
        for cell_idx, cell in voronoi_cells.iterrows():
            overlaps = []
            for parcel_idx, parcel_row in parcels_with_buildings.iterrows():
                intersection = cell.geometry.intersection(parcel_row.geometry)
                if not intersection.is_empty:
                    overlaps.append((parcel_idx, intersection.area))
            
            if overlaps:
                # Get the parcel with maximum overlap area
                max_overlap_parcel_idx = max(overlaps, key=lambda x: x[1])[0]
                # Get color directly from the mapping file (as string key)
                max_parcel_color = color_mapping.get(str(max_overlap_parcel_idx))
                if max_parcel_color:
                    cell_colors[cell_idx] = max_parcel_color
                else:
                    # Fallback to the color in the parcels dataframe if missing in mapping
                    cell_colors[cell_idx] = parcels_with_buildings.loc[max_overlap_parcel_idx, 'color']
        
        # Apply the colors to Voronoi cells
        for cell_idx, color in cell_colors.items():
            voronoi_cells.loc[cell_idx, 'color'] = color
        
        # Ensure all cells have a color (fallback to initial colors for any missing)
        if 'color' not in voronoi_cells.columns or voronoi_cells['color'].isna().any():
            missing_colors = voronoi_cells[voronoi_cells['color'].isna()].index if 'color' in voronoi_cells.columns else voronoi_cells.index
            for idx in missing_colors:
                voronoi_cells.loc[idx, 'color'] = random_hex_color(int(idx))
        
        # Render Voronoi image.
        voronoi_output = os.path.join(output_dirs['voronoi'], f"voronoi_{i:06d}.jpg")
        render_map(voronoi_cells.geometry.tolist(), voronoi_cells['color'].tolist(), area_geometry, voronoi_output)
        
        if generate_buildings:
            building_fragments = []
            fragment_colors = []
            for idx, building_row in all_buildings.iterrows():
                fragments = split_building_by_parcels(building_row.geometry, parcels_with_buildings)
                for frag, col in fragments:
                    building_fragments.append(frag)
                    fragment_colors.append(col)
            if not building_fragments:
                print(f"Sample {i}: No building fragments produced, skipping building image.")
            else:
                voronoi_buildings_output = os.path.join(output_dirs['voronoi_buildings'], f"voronoi_buildings_{i:06d}.jpg")
                render_map(building_fragments, fragment_colors, area_geometry, voronoi_buildings_output)
        
        print(f"Sample {i} processed successfully.")
    except Exception as e:
        print(f"Error processing sample {i} (parcel index {parcel_idx}): {str(e)}")

def generate_brooklyn_voronoi(buildings_path=BUILDINGS_PATH, parcels_path=PARCELS_PATH, blocks_path=BLOCKS_PATH,
                              sample_file=SAMPLE_FILE, output_dir=OUTPUT_DIR, num_samples=NUM_SAMPLES, 
                              generate_buildings=False, color_mapping_file=DEFAULT_COLOR_MAPPING_FILE):
    """
    Main function to generate Voronoi tessellations over a set of samples.
    Uses the same buffered parcel area as the ground truth script for image extent.
    If block clipping is enabled, Voronoi cells are clipped to block boundaries (within the buffered area).
    Voronoi cell colors are assigned via a spatial join and then overridden using the provided
    parcel-to-color mapping file.
    If --with-buildings is specified, a separate building overlay output is created where buildings
    are split by parcel boundaries so that each fragment receives the correct color.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(VORONOI_DIR, exist_ok=True)
    if generate_buildings:
        os.makedirs(VORONOI_BUILDINGS_DIR, exist_ok=True)
    
    # Verify color mapping file exists
    if not os.path.exists(color_mapping_file):
        print(f"Error: Color mapping file {color_mapping_file} not found.")
        print("Please run the parcels-buildings_ground-truth.py script first to generate the mapping.")
        sys.exit(1)
    
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
    
    # Load color mapping
    try:
        with open(color_mapping_file, 'r') as f:
            color_mapping = json.load(f)
        print(f"Loaded color mapping for {len(color_mapping)} parcels from {color_mapping_file}")
    except Exception as e:
        print(f"Error loading color mapping: {e}")
        print("Generating new random colors for parcels")
        color_mapping = {str(idx): random_hex_color(int(idx)) for idx in parcels_df.index}
        # Save the new color mapping
        with open(color_mapping_file, 'w') as f:
            json.dump(color_mapping, f, indent=2)
        print(f"Generated and saved new color mapping for {len(color_mapping)} parcels")
    
    blocks_df = None
    if USE_BLOCKS:
        blocks_df = get_blocks(parcels_df, blocks_path)
    
    if os.path.exists(sample_file):
        try:
            sampled_indices = np.load(sample_file)
            print(f"Loaded {len(sampled_indices)} sample indices from {sample_file}")
        except Exception as e:
            print(f"Error loading sample file: {e}")
            sampled_indices = None
    else:
        sampled_indices = None
        
    if sampled_indices is None:
        print("Generating new sample indices")
        if len(parcels_df) >= num_samples:
            sampled_indices = np.random.choice(parcels_df.index, num_samples, replace=False)
        else:
            sampled_indices = parcels_df.index.tolist()
            print(f"Warning: Only {len(parcels_df)} parcels available, using all.")
        np.save(sample_file, sampled_indices)
        print(f"Saved {len(sampled_indices)} sample indices to {sample_file}")
    
    total_samples = len(sampled_indices)
    output_dirs = {'voronoi': VORONOI_DIR}
    if generate_buildings:
        output_dirs['voronoi_buildings'] = VORONOI_BUILDINGS_DIR
    
    for start in tqdm(range(0, total_samples, BATCH_SIZE), desc="Processing batches"):
        end = min(start + BATCH_SIZE, total_samples)
        batch = sampled_indices[start:end]
        
        # Check for keyboard interrupt more gracefully
        try:
            for offset, parcel_idx in tqdm(enumerate(batch), total=len(batch), desc="Processing samples in batch", leave=False):
                process_sample(start + offset, parcel_idx, parcels_df, buildings_df, blocks_df,
                            output_dirs=output_dirs, generate_buildings=generate_buildings,
                            color_mapping=color_mapping)
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Saving progress...")
            checkpoint_file = os.path.join(output_dir, "last_processed_voronoi_index.txt")
            with open(checkpoint_file, 'w') as f:
                f.write(str(start + offset if 'offset' in locals() else start))
            print(f"Checkpoint saved up to sample index {start + offset if 'offset' in locals() else start}.")
            break
            
        gc.collect()
        checkpoint_file = os.path.join(output_dir, "last_processed_voronoi_index.txt")
        with open(checkpoint_file, 'w') as f:
            f.write(str(end))
        print(f"Checkpoint saved up to sample index {end}.")
    
    print("Completed generating Voronoi tessellations.")
    out_msg = f"Outputs saved to: {VORONOI_DIR}"
    if generate_buildings:
        out_msg += f" and {VORONOI_BUILDINGS_DIR}"
    print(out_msg)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Brooklyn Voronoi Tessellations with Consistent Zoom, Block Clipping, and Matching Parcel Colors")
    parser.add_argument('--buildings', type=str, default=BUILDINGS_PATH, help='Path to buildings shapefile')
    parser.add_argument('--parcels', type=str, default=PARCELS_PATH, help='Path to parcels shapefile')
    parser.add_argument('--blocks', type=str, default=BLOCKS_PATH, help='Path to blocks shapefile')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR, help='Base output directory')
    parser.add_argument('--sample-file', type=str, default=SAMPLE_FILE, help='Path to sample indices file')
    parser.add_argument('--num-samples', type=int, default=NUM_SAMPLES, help='Number of samples to process')
    parser.add_argument('--color-mapping', type=str, default=DEFAULT_COLOR_MAPPING_FILE, help='Path to parcel-to-color mapping JSON file')
    parser.add_argument('--with-buildings', action='store_true', help='Generate voronoi_buildings output')
    
    args = parser.parse_args()
    generate_brooklyn_voronoi(
        buildings_path=args.buildings,
        parcels_path=args.parcels,
        blocks_path=args.blocks,
        sample_file=args.sample_file,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        generate_buildings=args.with_buildings,
        color_mapping_file=args.color_mapping
    )