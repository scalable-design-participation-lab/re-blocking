import random
import geopandas as gpd
import os
from dotenv import load_dotenv
load_dotenv()
from cartopy.io.img_tiles  import MapboxTiles
import cartopy.crs as ccrs
from tqdm import tqdm
import matplotlib.pyplot as plt

def random_hex_color(seed=False):
  if seed:
    random.seed(seed)
    r = random.randint(0, 255)
    random.seed(seed+1000)
    g = random.randint(0, 255)
    random.seed(seed+2000)
    b = random.randint(0, 255)
  else:
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
  return "#{:02x}{:02x}{:02x}".format(r, g, b)

def remove_duplicates(df):
    indexes_to_remove = []
    my_dict = {}
    for index, row in tqdm(df.iterrows()):
        if row['geometry'] not in my_dict:
            my_dict[row['geometry']] = 1
        else:
            indexes_to_remove.append(index)
    df = df.drop(indexes_to_remove)
    df.reset_index(drop=True, inplace=True)
    return df

# loop through the parcels and add them to the map
def add_geometries(ax, df_parcels, crs_epsg, random_color = False):
    for row in df_parcels.itertuples():
        geometry = row.geometry
        if random_color == True: color = random_hex_color(int(row.bin))
        else: color = row.color
        ax.add_geometries(geometry, crs = crs_epsg, facecolor=color) # for Lat/Lon data.
        
# subset the data frames based on a buffer
def subset(df, df_buildings, index, distance = 75):
    selected_feature = df.loc[index]
    geometry_buffer = selected_feature.geometry.buffer(distance)
    geometry_bounds = selected_feature.geometry.buffer(distance-70)

    return df[df.within(geometry_buffer)], df_buildings[df_buildings.within(geometry_buffer)], geometry_bounds.bounds
        
def map_maker(df_parcels, df_buildings, bounds, index, scale=10, feature_type='both', random_color=False, img_folder = None, epsg = None):
    access_token = os.environ.get('MAPBOX_ACCESS_TOKEN')
    tiler = MapboxTiles(access_token, 'satellite-v9')
    crs_epsg = ccrs.epsg('3857')

    mercator = tiler.crs

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=mercator)

    # change figure size of the subplot
    my_dpi=96
    fig.set_size_inches(7, 7)
    # fig.figsize = (512/my_dpi, 512/my_dpi), dpi=my_dpi

    # calculate the centroid and max distance of the bounds
    dist1 = bounds[2]-bounds[0]
    dist2 = bounds[3]-bounds[1]
    max_dist = max(dist1, dist2)/2

    # calculate the centroid of the bounds
    centroid_x = (bounds[2]+bounds[0])/2
    centroid_y = (bounds[3]+bounds[1])/2

    # bounds = df_parcels.total_bounds with offset to create same aspect ratio
    ax.set_extent([centroid_x-max_dist, centroid_x+max_dist, centroid_y-max_dist, centroid_y+max_dist], crs=ccrs.epsg('3857'))
    # ax.set_extent([centroid_x-max_dist, centroid_x+max_dist, centroid_y-max_dist, centroid_y+max_dist], crs=ccrs.epsg('3857'))

    # if feature_type == 'parcels': add_parcels(ax, df_parcels, crs_epsg)
    if feature_type == 'parcels': 
        add_geometries(ax, df_parcels, crs_epsg)
        # ax.add_geometries(df_buildings.geometry, crs = crs_epsg, facecolor='white', edgecolor='black', linewidth=2.5, alpha=1)
    if feature_type == 'parcels' and random_color == True:
        add_geometries(ax, df_parcels, crs_epsg, random_color=True)
    if feature_type == 'buildings': 
        add_geometries(ax, df_buildings, crs_epsg)
    if feature_type == 'buildings' and random_color == True:
        add_geometries(ax, df_buildings, crs_epsg, random_color=True)
    if feature_type == 'both' and random_color == True: 
        add_geometries(ax, df_buildings, crs_epsg, random_color=True)
        add_geometries(ax, df_parcels, crs_epsg, random_color=True)
    if  feature_type == 'both': 
        # ax.add_geometries(df_buildings.geometry, crs = crs_epsg, facecolor='black', edgecolor='white', linewidth=1.5, alpha=1)
        add_geometries(ax, df_buildings, crs_epsg, random_color=True)
        add_geometries(ax, df_parcels, crs_epsg)

    # add the Mapbox tiles to the axis at zoom level 10 (Mapbox has 23 zoom levels)
    ax.add_image(tiler, scale)

    # save the figure
    plt.savefig(img_folder + f'{feature_type}_{index}.jpg', bbox_inches='tight', pad_inches = 0, dpi = my_dpi)
    
    # close the figure
    plt.close(fig)

    # ax.coastlines('10m')
    plt.show()
    
def split_building(buildings_with_multiple_parcels, buildings_with_parcel_info, df_buildings, df_parcels, 
                   threshold_high = 0.75, threshold_low = 0.15, buildings_clean_path = './'):
    
    # Create an empty GeoDataFrame to hold the split buildings
    split_buildings = gpd.GeoDataFrame()

    # Initialize a counter for tracking progress
    count = 0

    # Create a dictionary to store information about parcels to be removed
    to_remove = {}

    # Iterate through buildings with multiple parcels
    print("Iterate through buildings in multiple parcels and split them. Splitting...")
    for building_id in tqdm(buildings_with_multiple_parcels):
        # Print progress every 1000 buildings
        # if count % 1000 == 0:
        #     print(count)
        
        # Increment the counter
        count += 1
        
        # Select the building with the current building_id
        building = df_buildings[df_buildings["building_id"] == building_id].copy()
        
        # Get the list of parcel IDs that intersect with the building
        parcel_ids = buildings_with_parcel_info[buildings_with_parcel_info["building_id"] == building_id]["parcel_id"].tolist()

        # Initialize lists to store split geometries and areas
        split_geometries = []
        areas = []
        
        # Split the building geometry into separate parts based on the parcels
        i = 0
        for parcel_id in parcel_ids:
            split_geometries.append(df_parcels[df_parcels["parcel_id"] == parcel_id].geometry.intersection(building.geometry.unary_union)) 
            areas.append(split_geometries[i].area.values[0])
            i += 1
            
        # Calculate the normalized areas for each parcel
        areas_normalized = [a/sum(areas) for a in areas]
        
        # Find the maximum normalized area value and its index
        max_value = max(areas_normalized)
        max_index = areas_normalized.index(max_value)
        
        # If the maximum area is greater than or equal to threshold_high, remove everything except the one with high area
        if max_value >= threshold_high:
            to_remove[building_id] = parcel_ids[:max_index] + parcel_ids[max_index + 1:]
        else:
            # If the maximum area is less than threshold_high, remove all parcels below 0.15 and between 0.15 and threshold_high
            for i in range(len(areas_normalized)):
                if areas_normalized[i] <= threshold_low:
                    if building_id in to_remove:
                        to_remove[building_id] = to_remove[building_id] + [parcel_ids[i]]
                    else:
                        to_remove[building_id] = [parcel_ids[i]]
            for i in range(len(areas_normalized)):
                if threshold_low < areas_normalized[i] < threshold_high:
                    if building_id in to_remove:
                        to_remove[building_id] = to_remove[building_id] + [parcel_ids[i]]
                    else:
                        to_remove[building_id] = [parcel_ids[i]]
                    
                    # Create a temporary building with split geometry and update its attributes
                    building_temp = buildings_with_parcel_info[(buildings_with_parcel_info["building_id"] == building_id) & (buildings_with_parcel_info["parcel_id"] == parcel_ids[i])].copy()
                    building_temp.geometry = split_geometries[i].values
                    building_temp['building_id'] = str(building_temp['building_id'].values[0]) + '_' + str(i)
                    
                    # Append the temporary building to the split_buildings GeoDataFrame
                    split_buildings = split_buildings.append(building_temp, ignore_index=True)
                    
    buildings_with_parcel_info_new = buildings_with_parcel_info.copy()
    print("Add the new buildings' geometries...")
    for key, values in tqdm(to_remove.items()):
        for v in values:
            buildings_with_parcel_info_new = buildings_with_parcel_info_new[(buildings_with_parcel_info_new['building_id']!=key) | (buildings_with_parcel_info_new['parcel_id']!=v) ]
    # print(buildings_with_parcel_info_new.shape[0])
    # add new ones
    buildings_with_parcel_info_new = buildings_with_parcel_info_new.append(split_buildings, ignore_index=True)
    buildings_with_parcel_info_new['building_id'] = buildings_with_parcel_info_new['building_id'].astype(str)

    #save geopandas
    # buildings_with_parcel_info_new.to_file(buildings_clean_path, driver='GeoJSON')
    buildings_with_parcel_info_new.to_file(buildings_clean_path, driver='ESRI Shapefile')
    
    return buildings_with_parcel_info_new


def generate_dataset_specs(buildings_with_parcel_info, buildings_with_parcel_info_new, df_parcels):
    print('Generating dataset specs...')
    
    # Calculate number of buildings before/after split
    buildings_before = buildings_with_parcel_info.shape[0]
    buildings_after = buildings_with_parcel_info_new.shape[0]
    number_of_parcels = df_parcels.shape[0]
    
    try:
        df_parcels.drop(columns = ['index_left'], inplace=True)
    except:
        pass
    try:
        df_parcels.drop(columns = ['index_right'], inplace=True)
    except:
        pass
    try:
        buildings_with_parcel_info_new.drop(columns = ['index_left'], inplace=True)
    except:
        pass
    try:
        buildings_with_parcel_info_new.drop(columns = ['index_right'], inplace=True)
    except:
        pass
       
    # Spatial join to associate each building with a parcel
    joined_df = gpd.sjoin(buildings_with_parcel_info_new, df_parcels, op='within', how='left')
    
    # Group by parcel ID and count buildings
    building_counts_per_parcel = joined_df.groupby('index_right').size()
    
    # Filter parcels with more than one building
    parcels_with_multiple_buildings = building_counts_per_parcel[building_counts_per_parcel > 1]
    num_parcels_with_multiple_buildings = len(parcels_with_multiple_buildings)
    
    # Calculate parcel areas
    df_parcels['area'] = df_parcels.geometry.area
    
    # Plot histogram of parcel areas
    plt.figure(figsize=(10, 6))
    plt.hist(df_parcels['area'], bins=50, color='skyblue', edgecolor='black')
    plt.title('Parcel Area Distribution')
    plt.xlabel('Area (square meters)')
    plt.ylabel('Number of Parcels')
    plt.grid(True)
    plt.savefig('./dataset_specs/parcel_area_distribution.png', dpi=300)
    
    # Generate descriptive statistics for the 'area' column
    area_description = df_parcels['area'].describe()
    # area_description.to_csv('./dataset_specs/parcel_area_statistics.csv')
    
    # Spatial join to associate buildings with parcels for coverage calculation
    joined_df = gpd.sjoin(buildings_with_parcel_info_new, df_parcels, how='inner', op='intersects')
    joined_df['intersection_area'] = joined_df.apply(lambda row: row['geometry'].intersection(df_parcels.loc[row['index_right']].geometry).area, axis=1)
    building_area_per_parcel = joined_df.groupby('index_right')['intersection_area'].sum()
    
    # Calculate parcel areas and coverage percentage
    df_parcels['parcel_area'] = df_parcels.geometry.area
    df_parcels = df_parcels.join(building_area_per_parcel, on=df_parcels.index, rsuffix='_building')
    df_parcels['building_area'] = df_parcels['intersection_area'].fillna(0)
    df_parcels['coverage_percentage'] = (df_parcels['building_area'] / df_parcels['parcel_area']) * 100
    
    # Generate descriptive statistics for the 'coverage_percentage' column
    coverage_description = df_parcels['coverage_percentage'].describe()
    # coverage_description.to_csv('./dataset_specs/parcel_coverage_statistics.csv')
    
    with open('./dataset_specs/dataset_summary_and_statistics.txt', 'w') as file:
        # Save initial summary information
        file.write(f'Buildings before: {buildings_before}\n')
        file.write(f'Buildings after: {buildings_after}\n')
        file.write(f'Number of parcels: {number_of_parcels}\n')
        file.write(f'Parcels with multiple buildings: {num_parcels_with_multiple_buildings}\n\n')
        
        # Save area statistics
        file.write('Area Statistics:\n')
        area_stats = area_description.to_string()  # Convert the area description DataFrame/Series to a string
        file.write(area_stats)
        file.write('\n\n')  # Add some space before the next section
        
        # Save coverage statistics
        file.write('Coverage Statistics:\n')
        coverage_stats = coverage_description.to_string()  # Convert the coverage description DataFrame/Series to a string
        file.write(coverage_stats)