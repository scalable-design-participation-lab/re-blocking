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
BLOCK_VORONOI_DIR = os.path.join(OUTPUT_DIR, "voronoi-block")
BLOCK_WITH_BUILDINGS_DIR = os.path.join(OUTPUT_DIR, "voronoi-block-buildings")
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

def identify_blocks_from_parcels(parcels_gdf):
    """
    Identify blocks from parcel data by grouping adjacent parcels.
    This uses a spatial join approach to group adjacent parcels.
    """
    print("Identifying blocks from parcel data...")
    
    # Create a small buffer around each parcel to identify adjacency
    parcels_buffered = parcels_gdf.copy()
    parcels_buffered['geometry'] = parcels_gdf.geometry.buffer(0.1)
    
    # Create a spatial index to speed up operations
    spatial_index = parcels_buffered.sindex
    
    # Initialize block ID for each parcel
    parcels_gdf['block_id'] = None
    
    # Keep track of processed parcels
    processed = set()
    next_block_id = 1
    
    for idx, parcel in tqdm_base(parcels_gdf.iterrows(), total=len(parcels_gdf), desc="Grouping parcels into blocks"):
        if idx in processed:
            continue
            
        # Start a new block
        current_block = set([idx])
        parcels_gdf.loc[idx, 'block_id'] = next_block_id
        processed.add(idx)
        
        # Find all adjacent parcels recursively
        to_process = set([idx])
        while to_process:
            current = to_process.pop()
            buffer_geom = parcels_buffered.loc[current, 'geometry']
            
            # Get possible intersections using spatial index
            possible_matches_idx = list(spatial_index.intersection(buffer_geom.bounds))
            possible_matches = parcels_gdf.iloc[possible_matches_idx]
            
            # Check which ones actually intersect
            for match_idx, match in possible_matches.iterrows():
                if match_idx != current and match_idx not in processed:
                    if buffer_geom.intersects(parcels_buffered.loc[match_idx, 'geometry']):
                        parcels_gdf.loc[match_idx, 'block_id'] = next_block_id
                        current_block.add(match_idx)
                        to_process.add(match_idx)
                        processed.add(match_idx)
        
        next_block_id += 1
    
    print(f"Identified {next_block_id-1} blocks from {len(parcels_gdf)} parcels")
    return parcels_gdf

def extract_block_geometries(parcels_with_blocks):
    """
    Extract block geometries from parcels with block IDs.
    Returns a GeoDataFrame with one row per block.
    """
    # Group by block_id and dissolve to get block geometries
    blocks = parcels_with_blocks.dissolve(by='block_id', aggfunc='first')
    blocks = blocks.reset_index()
    
    print(f"Created {len(blocks)} block geometries")
    return blocks

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
    # First try momepy, which handles complex cases better
    try:
        # Create a unique ID for each building
        buildings_with_id = buildings_gdf.copy()
        buildings_with_id['unique_id'] = range(len(buildings_with_id))
        
        # Generate the tessellation with momepy
        tessellation = momepy.Tessellation(
            gdf=buildings_with_id, 
            unique_id='unique_id',
            enclosures=gpd.GeoDataFrame(geometry=[boundary_geometry]),
            enclosure_id=0
        ).tessellation
        
        # Add colors
        tessellation['color'] = [random_hex_color(i) for i in range(len(tessellation))]
        
        return tessellation
    except Exception as e:
        print(f"Momepy failed, trying manual method: {e}")
        return manual_voronoi_tessellation(buildings_gdf, boundary_geometry)

def render_voronoi_tessellation(voronoi_gdf, area_geometry, output_path, buffer_distance=BUFFER_DISTANCE, 
                              buildings_gdf=None, include_buildings=False, show_boundary=False):
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
    
    # Show boundary if requested
    if show_boundary:
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

def process_batch(sampled_indices_batch, buildings_df, parcels_df, blocks_gdf, citywide_voronoi, 
                 start_idx, output_types, output_dirs, buffer_distance):
    """
    Process a batch of samples to manage memory consumption.
    
    Args:
        sampled_indices_batch: List of indices to process
        buildings_df: GeoDataFrame of buildings
        parcels_df: GeoDataFrame of parcels
        blocks_gdf: GeoDataFrame of blocks derived from parcels
        citywide_voronoi: Pre-computed citywide Voronoi tessellation
        start_idx: Starting index for this batch
        output_types: List of output types to generate ('block', 'buffer', 'citywide')
        output_dirs: Dictionary of output directories
        buffer_distance: Distance for buffer-based Voronoi
    """
    for batch_i, idx in enumerate(tqdm_base(sampled_indices_batch, desc=f"Processing batch {start_idx}-{start_idx+len(sampled_indices_batch)-1}")):
        i = start_idx + batch_i
        
        # Get the specific parcel
        try:
            parcel = parcels_df.loc[idx]
        except:
            print(f"Error accessing parcel at index {idx}")
            continue
        
        # ------------------------------------------------------------
        # Method 1: Block-based Voronoi (NEW)
        # ------------------------------------------------------------
        if 'block' in output_types:
            # Find which block this parcel belongs to
            block_id = parcel.get('block_id')
            
            if block_id is not None:
                # Get the block geometry
                block_geom = blocks_gdf[blocks_gdf['block_id'] == block_id].geometry.iloc[0]
                
                # Get buildings within this block
                buildings_in_block = buildings_df[buildings_df.geometry.within(block_geom)]
                
                if len(buildings_in_block) > 0:
                    # Generate Voronoi tessellation within block
                    block_voronoi = generate_voronoi_tessellation(buildings_in_block, block_geom)
                    
                    if not block_voronoi.empty:
                        # Generate output filename for block method
                        block_output_path = os.path.join(output_dirs['block'], f"voronoi-block_{i:06d}.jpg")
                        
                        # Render and save block-based Voronoi
                        render_voronoi_tessellation(
                            block_voronoi,
                            block_geom,
                            block_output_path,
                            buildings_gdf=None,
                            include_buildings=False,
                            show_boundary=True
                        )
                        
                        # Generate output filename for block method with buildings
                        block_buildings_path = os.path.join(output_dirs['block_buildings'], f"voronoi-block-buildings_{i:06d}.jpg")
                        
                        # Render and save block-based Voronoi with buildings
                        render_voronoi_tessellation(
                            block_voronoi,
                            block_geom,
                            block_buildings_path,
                            buildings_gdf=buildings_in_block,
                            include_buildings=True,
                            show_boundary=True
                        )
        
        # ------------------------------------------------------------
        # Method 2: Buffer-based Voronoi (ORIGINAL)
        # ------------------------------------------------------------
        if 'buffer' in output_types:
            # Create buffer around parcel
            buffer_geometry = parcel.geometry.buffer(buffer_distance * 0.75)
            
            # Get buildings within buffer
            buildings_in_buffer = buildings_df[buildings_df.geometry.within(buffer_geometry)]
            
            if len(buildings_in_buffer) > 0:
                # Generate Voronoi tessellation within buffer
                buffer_voronoi = generate_voronoi_tessellation(buildings_in_buffer, buffer_geometry)
                
                if not buffer_voronoi.empty:
                    # Generate output filename for buffer method
                    buffer_output_path = os.path.join(output_dirs['buffer'], f"voronoi-buffer_{i:06d}.jpg")
                    
                    # Render and save buffer-based Voronoi
                    render_voronoi_tessellation(
                        buffer_voronoi,
                        buffer_geometry,
                        buffer_output_path,
                        buffer_distance=buffer_distance,
                        buildings_gdf=None,
                        include_buildings=False,
                        show_boundary=True
                    )
                    
                    # Generate output filename for buffer method with buildings
                    buffer_buildings_path = os.path.join(output_dirs['buffer_buildings'], f"voronoi-buffer-buildings_{i:06d}.jpg")
                    
                    # Render and save buffer-based Voronoi with buildings
                    render_voronoi_tessellation(
                        buffer_voronoi,
                        buffer_geometry,
                        buffer_buildings_path,
                        buffer_distance=buffer_distance,
                        buildings_gdf=buildings_in_buffer,
                        include_buildings=True,
                        show_boundary=True
                    )
        
        # ------------------------------------------------------------
        # Method 3: Citywide Voronoi (clipped to view area)
        # ------------------------------------------------------------
        if 'citywide' in output_types:
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
                    citywide_output_path = os.path.join(output_dirs['citywide'], f"voronoi_{i:06d}.jpg")
                    
                    # Get buildings in the view area
                    buildings_in_view = buildings_df[buildings_df.geometry.within(view_geometry)]
                    
                    # Render and save citywide-based Voronoi
                    render_voronoi_tessellation(
                        citywide_clipped,
                        view_geometry,
                        citywide_output_path,
                        buffer_distance=buffer_distance,
                        buildings_gdf=None,
                        include_buildings=False,
                        show_boundary=False
                    )
                    
                    # Generate output filename for citywide method with buildings
                    citywide_buildings_path = os.path.join(output_dirs['citywide_buildings'], f"voronoi-buildings_{i:06d}.jpg")
                    
                    # Render and save citywide-based Voronoi with buildings
                    render_voronoi_tessellation(
                        citywide_clipped,
                        view_geometry,
                        citywide_buildings_path,
                        buffer_distance=buffer_distance,
                        buildings_gdf=buildings_in_view,
                        include_buildings=True,
                        show_boundary=False
                    )
        
        # Force garbage collection every few samples to prevent memory buildup
        if batch_i % 5 == 0:
            gc.collect()

def generate_brooklyn_voronoi(buildings_path=BUILDINGS_PATH, parcels_path=PARCELS_PATH, 
                             output_dir=OUTPUT_DIR, sample_file=SAMPLE_FILE, buffer_distance=BUFFER_DISTANCE,
                             start_from=0, output_types=None):
    """
    Generate Voronoi tessellations using block, buffer, and/or citywide approaches.
    
    Args:
        buildings_path: Path to building shapefile
        parcels_path: Path to parcels shapefile
        output_dir: Base directory for outputs
        sample_file: File with sample indices
        buffer_distance: Distance for buffer-based Voronoi
        start_from: Index to start processing from (0-based)
        output_types: List of output types to generate ('block', 'buffer', 'citywide')
                     If None, defaults to ['block', 'buffer']
    """
    # Default to block-based output if none specified
    if output_types is None:
        output_types = ['block', 'buffer']
    
    # Set up output directories
    output_dirs = {
        'block': os.path.join(output_dir, "voronoi-block"),
        'block_buildings': os.path.join(output_dir, "voronoi-block-buildings"),
        'buffer': os.path.join(output_dir, "voronoi-buffer"),
        'buffer_buildings': os.path.join(output_dir, "voronoi-buffer-buildings"),
        'citywide': os.path.join(output_dir, "voronoi"),
        'citywide_buildings': os.path.join(output_dir, "voronoi-buildings")
    }
    
    # Create only the needed output directories
    for output_type in output_types:
        if output_type == 'block':
            os.makedirs(output_dirs['block'], exist_ok=True)
            os.makedirs(output_dirs['block_buildings'], exist_ok=True)
        elif output_type == 'buffer':
            os.makedirs(output_dirs['buffer'], exist_ok=True)
            os.makedirs(output_dirs['buffer_buildings'], exist_ok=True)
        elif output_type == 'citywide':
            os.makedirs(output_dirs['citywide'], exist_ok=True)
            os.makedirs(output_dirs['citywide_buildings'], exist_ok=True)
    
    # Load datasets and convert to web mercator projection
    print(f"Loading building data from: {buildings_path}")
    try:
        if not os.path.exists(buildings_path):
            print(f"Error: Building shapefile does not exist at {buildings_path}")
            return
        buildings_df = gpd.read_file(buildings_path)
        print(f"Successfully loaded {len(buildings_df)} buildings")
        buildings_df = buildings_df.to_crs(3857)
    except Exception as e:
        print(f"Error loading building data: {str(e)}")
        return
    
    print(f"Loading parcel data from: {parcels_path}")
    try:
        if not os.path.exists(parcels_path):
            print(f"Error: Parcel shapefile does not exist at {parcels_path}")
            return
        parcels_df = gpd.read_file(parcels_path)
        print(f"Successfully loaded {len(parcels_df)} parcels")
        parcels_df = parcels_df.to_crs(3857)
    except Exception as e:
        print(f"Error loading parcel data: {str(e)}")
        return
    
    # Check if sample file exists
    if not os.path.exists(sample_file):
        print(f"Error: Sample file {sample_file} not found.")
        print("Please run the ground truth generator script first.")
        return
    
    # Load sample indices
    try:
        sampled_indices = np.load(sample_file)
        print(f"Loaded {len(sampled_indices)} sample indices from {sample_file}")
    except Exception as e:
        print(f"Error loading sample file: {str(e)}")
        return
    
    # Identify blocks from parcels (for block-based approach)
    blocks_gdf = None
    if 'block' in output_types:
        parcels_with_blocks = identify_blocks_from_parcels(parcels_df)
        blocks_gdf = extract_block_geometries(parcels_with_blocks)
        # Update parcels_df with block_id
        parcels_df = parcels_with_blocks
    
    # Generate citywide Voronoi tessellation once for all samples (if needed)
    citywide_voronoi = None
    if 'citywide' in output_types:
        print("Generating citywide Voronoi tessellation...")
        try:
            # Use convex hull of all parcels as boundary
            city_boundary = parcels_df.unary_union.convex_hull
            
            # Generate the citywide tessellation
            citywide_voronoi = generate_voronoi_tessellation(buildings_df, city_boundary)
            print(f"Generated citywide Voronoi with {len(citywide_voronoi)} cells")
            
            # Set random colors for citywide Voronoi cells
            citywide_voronoi['color'] = [random_hex_color(i) for i in range(len(citywide_voronoi))]
        except Exception as e:
            print(f"Error generating citywide Voronoi: {e}")
            if 'citywide' in output_types:
                print("Removing 'citywide' from output types")
                output_types.remove('citywide')
    
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
            blocks_gdf,
            citywide_voronoi,
            start_idx,
            output_types,
            output_dirs,
            buffer_distance
        )
        
        # Force garbage collection between batches
        gc.collect()
        print(f"Completed batch {start_idx}-{end_idx-1} of {num_samples}")
        
        # Save a checkpoint file with the current index
        checkpoint_file = os.path.join(output_dir, "last_processed_index.txt")
        with open(checkpoint_file, 'w') as f:
            f.write(str(end_idx))
        print(f"Checkpoint saved: processed up to index {end_idx-1}")
    
    print(f"Completed generating Voronoi tessellations.")
    print(f"Outputs saved to: {output_dir}")

#########################################################
# MAIN EXECUTION
#########################################################

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Generate Voronoi tessellations clipped by block boundaries')
    parser.add_argument('--start', type=int, default=0, help='Starting index for processing')
    parser.add_argument('--output-types', type=str, default='block,buffer', 
                       help='Comma-separated list of output types to generate (block,buffer,citywide)')
    parser.add_argument('--buildings', type=str, default=BUILDINGS_PATH, help='Path to buildings shapefile')
    parser.add_argument('--parcels', type=str, default=PARCELS_PATH, help='Path to parcels shapefile')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR, help='Base directory for outputs')
    parser.add_argument('--sample-file', type=str, default=SAMPLE_FILE, help='File with sample indices')
    parser.add_argument('--buffer-distance', type=float, default=BUFFER_DISTANCE, help='Distance for buffer-based Voronoi')
    args = parser.parse_args()
    
    # Convert string paths to Path objects for better cross-platform compatibility
    buildings_path = Path(args.buildings)
    parcels_path = Path(args.parcels)
    output_dir = Path(args.output_dir)
    sample_file = Path(args.sample_file)
    
    # Parse output types - now include buffer by default
    output_types = args.output_types.split(',')
    print(f"Generating Voronoi tessellations for types: {output_types}")
    
    # Check if there's a checkpoint file to resume from
    checkpoint_file = output_dir / "last_processed_index.txt"
    start_index = args.start
    
    if checkpoint_file.exists() and start_index == 0:
        with open(checkpoint_file, 'r') as f:
            saved_index = int(f.read().strip())
            print(f"Found checkpoint file. Last processed index was {saved_index-1}")
            start_index = saved_index
    
    # Set lower memory footprint for matplotlib
    plt.rcParams['savefig.dpi'] = DPI
    plt.rcParams['figure.max_open_warning'] = 10
    
    print(f"Starting processing from index {start_index}")
    
    # Generate Voronoi tessellations with specified output types
    generate_brooklyn_voronoi(
        buildings_path=buildings_path,
        parcels_path=parcels_path,
        output_dir=output_dir,
        sample_file=sample_file,
        buffer_distance=args.buffer_distance,
        start_from=start_index,
        output_types=output_types
    )