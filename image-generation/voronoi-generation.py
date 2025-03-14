import os
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, MultiPoint, box
from shapely.ops import voronoi_diagram, unary_union
import momepy
from tqdm import tqdm as tqdm_base
import random
import warnings
from pathlib import Path
import cartopy.crs as ccrs
from cartopy.io.img_tiles import MapboxTiles
from dotenv import load_dotenv
import gc  # For garbage collection
import argparse  # For command line arguments

# Load environment variables
load_dotenv()

# Suppress warnings
warnings.filterwarnings('ignore')

#########################################################
# CONFIGURATION - ADJUST PARAMETERS AS NEEDED
#########################################################

# Input Paths
BUILDINGS_PATH = "/home/ls/sites/re-blocking/data/shapefiles/ny-manhattan-buildings/geo_export_a80ea1a2-e8e0-4ffd-862c-1199433ac303.shp"
PARCELS_PATH = "/home/ls/sites/re-blocking/data/shapefiles/ny-manhattan-parcels/NYC_2021_Tax_Parcels_SHP_2203/Kings_2021_Tax_Parcels_SHP_2203.shp"

# Output Settings
OUTPUT_DIR = "brooklyn_comparison"
BUFFER_VORONOI_DIR = os.path.join(OUTPUT_DIR, "voronoi-buffer")
BUFFER_WITH_BUILDINGS_DIR = os.path.join(OUTPUT_DIR, "voronoi-buffer-buildings")
CITYWIDE_VORONOI_DIR = os.path.join(OUTPUT_DIR, "voronoi")
CITYWIDE_WITH_BUILDINGS_DIR = os.path.join(OUTPUT_DIR, "voronoi-buildings")
SAMPLE_FILE = os.path.join(OUTPUT_DIR, "brooklyn_samples.npy")

# Processing Parameters
BUFFER_DISTANCE = 200
RANDOM_SEED = 42
BATCH_SIZE = 50  # Smaller batch size to avoid memory issues

# Visualization Settings - Matching original script
FIGURE_SIZE = (7, 7)  # Matching original 7x7 inches
DPI = 96  # Matching original DPI of 96
SATELLITE_ZOOM = 18  # Matching original zoom level
EXTENT_SCALE_FACTOR = 0.7  # Adjusted for better balance

# Style Parameters
INCLUDE_EDGES = False  # Set to False to remove edge lines around parcels

#########################################################
# FUNCTIONS
#########################################################

def random_hex_color(seed=None):
    """Generate a random hex color, optionally with a seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def manual_voronoi_tessellation(buildings_gdf, boundary_geometry):
    """Manual Voronoi tessellation if momepy fails."""
    if len(buildings_gdf) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=buildings_gdf.crs)
    
    try:
        # Create centroids for all buildings
        centroids = [Point(geom.centroid) for geom in buildings_gdf.geometry]
        
        # Create a multipoint from all centroids
        multipoint = MultiPoint(centroids)
        
        # Generate Voronoi diagram
        voronoi_polys = voronoi_diagram(multipoint, envelope=boundary_geometry)
        
        # Extract polygons and clip with boundary
        clipped_polys = []
        for poly in voronoi_polys.geoms:
            try:
                clipped_poly = poly.intersection(boundary_geometry)
                if clipped_poly.area > 0:
                    clipped_polys.append(clipped_poly)
            except Exception:
                continue
        
        # Create GeoDataFrame
        voronoi_gdf = gpd.GeoDataFrame(geometry=clipped_polys, crs=buildings_gdf.crs)
        
        # Add colors
        voronoi_gdf['color'] = [random_hex_color(i) for i in range(len(voronoi_gdf))]
        
        return voronoi_gdf
    
    except Exception as e:
        print(f"Manual Voronoi also failed: {e}")
        return gpd.GeoDataFrame(geometry=[], crs=buildings_gdf.crs)

def generate_voronoi_tessellation(buildings_gdf, boundary_geometry):
    """Generate Voronoi tessellation based on building centroids."""
    # Skip momepy and just use the manual method directly
    return manual_voronoi_tessellation(buildings_gdf, boundary_geometry)

def render_voronoi_tessellation(voronoi_gdf, area_geometry, output_path, buffer_distance=BUFFER_DISTANCE, buildings_gdf=None, include_buildings=False, show_buffer_boundary=False):
    """
    Render Voronoi tessellation on satellite imagery with optional building footprints.
    """
    # Get Mapbox token from environment
    mapbox_token = os.environ.get('MAPBOX_ACCESS_TOKEN')
    if not mapbox_token:
        print("Warning: MAPBOX_ACCESS_TOKEN not found in environment variables")
        return
    
    # Create figure with fixed size to ensure consistency
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    
    # Use Mapbox satellite imagery
    tiler = MapboxTiles(mapbox_token, 'satellite-v9')
    ax = fig.add_subplot(1, 1, 1, projection=tiler.crs)
    
    # Set extent based on area geometry
    bounds = area_geometry.bounds
    
    # Calculate the centroid and max distance for square aspect ratio
    centroid_x = (bounds[2] + bounds[0]) / 2
    centroid_y = (bounds[3] + bounds[1]) / 2
    
    # Use consistent scaling approach
    if show_buffer_boundary:
        # For buffer version, use exact buffer radius to set extent
        buffer_radius = buffer_distance * 0.75
        
        ax.set_extent([
            centroid_x - buffer_radius,
            centroid_x + buffer_radius,
            centroid_y - buffer_radius, 
            centroid_y + buffer_radius
        ], crs=ccrs.epsg('3857'))
    else:
        # For non-buffer version, use standard approach
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
    
    # Add satellite imagery at specified zoom level
    ax.add_image(tiler, SATELLITE_ZOOM)
    
    # Add Voronoi cells - matching original script's styling
    for idx, row in voronoi_gdf.iterrows():
        if INCLUDE_EDGES:
            ax.add_geometries(
                [row.geometry], 
                crs=ccrs.epsg('3857'),
                facecolor=row['color'],
                edgecolor='white',
                linewidth=0.5,
                alpha=0.7 if include_buildings else 1.0
            )
        else:
            # Original script style - no edges
            ax.add_geometries(
                [row.geometry], 
                crs=ccrs.epsg('3857'),
                facecolor=row['color'],
                alpha=0.7 if include_buildings else 1.0
            )
    
    # Add building footprints if requested
    if include_buildings and buildings_gdf is not None:
        for idx, row in buildings_gdf.iterrows():
            ax.add_geometries(
                [row.geometry], 
                crs=ccrs.epsg('3857'),
                facecolor='black',
                edgecolor='white' if INCLUDE_EDGES else None,
                linewidth=0.3 if INCLUDE_EDGES else 0,
                alpha=0.8
            )
    
    # Show buffer boundary if requested
    if show_buffer_boundary:
        ax.add_geometries(
            [area_geometry],
            crs=ccrs.epsg('3857'),
            facecolor='none',
            edgecolor='white',
            linewidth=2.5,
            alpha=0.6
        )
    
    # Remove axes
    ax.set_axis_off()
    
    # Change file extension from .png to .jpg to match original script
    output_path = output_path.replace('.png', '.jpg')
    
    # Save figure with tight layout and no padding - matching original script
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=DPI)
    plt.close(fig)
    
    # Force garbage collection to prevent memory buildup
    gc.collect()

def create_view_box(center_point, width, height):
    """Create a rectangular view box around a center point."""
    return box(
        center_point.x - width/2,
        center_point.y - height/2,
        center_point.x + width/2,
        center_point.y + height/2
    )

def process_batch(sampled_indices_batch, buildings_df, parcels_df, citywide_voronoi, 
                 start_idx, buffer_dir, buffer_buildings_dir, citywide_dir, 
                 citywide_buildings_dir, buffer_distance):
    """Process a batch of samples to manage memory consumption."""
    for batch_i, idx in enumerate(tqdm_base(sampled_indices_batch, desc=f"Processing batch {start_idx}-{start_idx+len(sampled_indices_batch)-1}")):
        i = start_idx + batch_i
        
        # Get the specific parcel
        try:
            parcel = parcels_df.loc[idx]
        except:
            print(f"Error accessing parcel at index {idx}")
            continue
        
        # ------------------------------------------------------------
        # Method 1: Buffer-based Voronoi
        # ------------------------------------------------------------
        
        # Create buffer around parcel
        buffer_geometry = parcel.geometry.buffer(buffer_distance * 0.75)
        
        # Get buildings within buffer
        buildings_in_buffer = buildings_df[buildings_df.geometry.within(buffer_geometry)]
        
        if len(buildings_in_buffer) > 0:
            # Generate Voronoi tessellation within buffer
            buffer_voronoi = generate_voronoi_tessellation(buildings_in_buffer, buffer_geometry)
            
            if not buffer_voronoi.empty:
                # Generate output filename for buffer method
                buffer_output_path = os.path.join(buffer_dir, f"voronoi-buffer_{i:06d}.jpg")
                
                # Render and save buffer-based Voronoi
                render_voronoi_tessellation(
                    buffer_voronoi,
                    buffer_geometry,
                    buffer_output_path,
                    buffer_distance=buffer_distance,
                    buildings_gdf=buildings_in_buffer,
                    include_buildings=False,
                    show_buffer_boundary=True
                )
                
                # Generate output filename for buffer method with buildings
                buffer_buildings_path = os.path.join(buffer_buildings_dir, f"voronoi-buffer-buildings_{i:06d}.jpg")
                
                # Render and save buffer-based Voronoi with buildings
                render_voronoi_tessellation(
                    buffer_voronoi,
                    buffer_geometry,
                    buffer_buildings_path,
                    buffer_distance=buffer_distance,
                    buildings_gdf=buildings_in_buffer,
                    include_buildings=True,
                    show_buffer_boundary=True
                )
        
        # ------------------------------------------------------------
        # Method 2: Citywide Voronoi (clipped to view area)
        # ------------------------------------------------------------
        
        # Create a view box around the parcel
        view_geometry = parcel.geometry.buffer(buffer_distance * 0.75)
        
        # Extract Voronoi cells that intersect with view area
        voronoi_in_view = citywide_voronoi[citywide_voronoi.intersects(view_geometry)].copy()
        
        if not voronoi_in_view.empty:
            # Clip the cells to the view boundary
            clipped_cells = []
            for _, cell in voronoi_in_view.iterrows():
                try:
                    clipped_geom = cell.geometry.intersection(view_geometry)
                    if clipped_geom.area > 0:
                        clipped_cells.append({
                            'geometry': clipped_geom,
                            'color': cell['color']
                        })
                except Exception as e:
                    continue
            
            # Create GeoDataFrame from clipped cells
            citywide_clipped = gpd.GeoDataFrame(clipped_cells, crs=buildings_df.crs)
            
            if not citywide_clipped.empty:
                # Generate output filename for citywide method - with consistent naming
                citywide_output_path = os.path.join(citywide_dir, f"voronoi_{i:06d}.jpg")
                
                # Get buildings in the view area
                buildings_in_view = buildings_df[buildings_df.geometry.within(view_geometry)]
                
                # Render and save citywide-based Voronoi
                render_voronoi_tessellation(
                    citywide_clipped,
                    view_geometry,
                    citywide_output_path,
                    buffer_distance=buffer_distance,
                    buildings_gdf=buildings_in_view,
                    include_buildings=False,
                    show_buffer_boundary=False
                )
                
                # Generate output filename for citywide method with buildings
                citywide_buildings_path = os.path.join(citywide_buildings_dir, f"voronoi-buildings_{i:06d}.jpg")
                
                # Render and save citywide-based Voronoi with buildings
                render_voronoi_tessellation(
                    citywide_clipped,
                    view_geometry,
                    citywide_buildings_path,
                    buffer_distance=buffer_distance,
                    buildings_gdf=buildings_in_view,
                    include_buildings=True,
                    show_buffer_boundary=False
                )
        
        # Force garbage collection every few samples to prevent memory buildup
        if batch_i % 5 == 0:
            gc.collect()

def generate_brooklyn_voronoi(buildings_path=BUILDINGS_PATH, parcels_path=PARCELS_PATH, 
                             buffer_dir=BUFFER_VORONOI_DIR, buffer_buildings_dir=BUFFER_WITH_BUILDINGS_DIR,
                             citywide_dir=CITYWIDE_VORONOI_DIR, citywide_buildings_dir=CITYWIDE_WITH_BUILDINGS_DIR,
                             sample_file=SAMPLE_FILE, buffer_distance=BUFFER_DISTANCE,
                             start_from=0):
    """
    Generate Voronoi tessellations using both buffer and citywide approaches.
    
    Args:
        start_from: Index to start processing from (0-based)
    """
    
    # Load datasets and convert to web mercator projection
    print("Loading building and parcel data...")
    buildings_df = gpd.read_file(buildings_path).to_crs(3857)
    parcels_df = gpd.read_file(parcels_path).to_crs(3857)
    
    # Create output directories
    os.makedirs(buffer_dir, exist_ok=True)
    os.makedirs(buffer_buildings_dir, exist_ok=True)
    os.makedirs(citywide_dir, exist_ok=True)
    os.makedirs(citywide_buildings_dir, exist_ok=True)
    
    # Check if sample file exists
    if not os.path.exists(sample_file):
        print(f"Error: Sample file {sample_file} not found.")
        print("Please run the ground truth generator script first.")
        return
    
    # Load sample indices
    sampled_indices = np.load(sample_file)
    print(f"Loaded {len(sampled_indices)} sample indices from {sample_file}")
    
    # Generate citywide Voronoi tessellation once for all samples
    print("Generating citywide Voronoi tessellation...")
    try:
        # Use convex hull of all parcels as boundary
        city_boundary = parcels_df.unary_union.convex_hull
        
        # Generate the citywide tessellation
        citywide_voronoi = generate_voronoi_tessellation(buildings_df, city_boundary)
        print(f"Generated citywide Voronoi with {len(citywide_voronoi)} cells")
        
        # Set random colors for citywide Voronoi cells
        citywide_voronoi['color'] = [random_hex_color(i) for i in range(len(citywide_voronoi))]
        
        # Process in batches to manage memory
        num_samples = len(sampled_indices)
        
        # Skip to the requested starting point
        if start_from > 0:
            print(f"Skipping first {start_from} samples as requested")
        
        # Process in smaller batches to avoid memory issues
        batch_size = BATCH_SIZE
        
        for start_idx in range(start_from, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = sampled_indices[start_idx:end_idx]
            
            process_batch(
                batch_indices, 
                buildings_df, 
                parcels_df, 
                citywide_voronoi,
                start_idx,
                buffer_dir, 
                buffer_buildings_dir, 
                citywide_dir, 
                citywide_buildings_dir,
                buffer_distance
            )
            
            # Force garbage collection between batches
            gc.collect()
            print(f"Completed batch {start_idx}-{end_idx-1} of {num_samples}")
            
            # Save a checkpoint file with the current index
            checkpoint_file = os.path.join(OUTPUT_DIR, "last_processed_index.txt")
            with open(checkpoint_file, 'w') as f:
                f.write(str(end_idx))
            print(f"Checkpoint saved: processed up to index {end_idx-1}")
    
    except Exception as e:
        print(f"Error during processing: {e}")
    
    print(f"Completed generating Voronoi tessellations.")
    print(f"Outputs saved to: \n  - Buffer-based: {buffer_dir}\n  - Buffer with buildings: {buffer_buildings_dir}\n  - Citywide: {citywide_dir}\n  - Citywide with buildings: {citywide_buildings_dir}")

#########################################################
# MAIN EXECUTION
#########################################################

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Generate Voronoi tessellations')
    parser.add_argument('--start', type=int, default=0, help='Starting index for processing')
    args = parser.parse_args()
    
    # Check if there's a checkpoint file to resume from
    checkpoint_file = os.path.join(OUTPUT_DIR, "last_processed_index.txt")
    start_index = args.start
    
    if os.path.exists(checkpoint_file) and start_index == 0:
        with open(checkpoint_file, 'r') as f:
            saved_index = int(f.read().strip())
            print(f"Found checkpoint file. Last processed index was {saved_index-1}")
            start_index = saved_index
    
    # Set lower memory footprint for matplotlib
    plt.rcParams['savefig.dpi'] = DPI
    plt.rcParams['figure.max_open_warning'] = 10
    
    print(f"Starting processing from index {start_index}")
    
    # Generate both types of Voronoi tessellations
    generate_brooklyn_voronoi(start_from=start_index)