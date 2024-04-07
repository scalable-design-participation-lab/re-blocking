# import libraries
import geopandas as gpd
import os
from dotenv import load_dotenv
load_dotenv(override=True) # loads .env file from main folder for Mapbox API key and local project path
import random

from helper import *

import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='This is a script to generate two image datasets, parcel_images and one for building_images, from the respective shp or geoJSON data.')

    # Add command-line arguments
    parser.add_argument("--buildings_path", help ='Path to buildings (shapefile or geoJSON)', type=str, required=False, default='./data/AlleghenyCounty_Footprints.geojson')
    parser.add_argument("--parcels_path", help ='Path to parcels (shapefile or geoJSON)', type=str, required=False, default='./data/AlleghenyCounty_Parcels.geojson')
    parser.add_argument("--blocks_path", help ='Path to blocks (shapefile or geoJSON)', type=str, required=False, default='./data/blocks')
    parser.add_argument("--split_buildings", help ='Whether to split buildings', type=bool, required=False, default=False)
    parser.add_argument("--load_split_buildings", help ='Whether to load split buildings', type=bool, required=False, default=True)
    parser.add_argument("--threshold_high", help = (
            "A decimal value between 0 and 1 representing the high building-parcel overlap threshold. "
            "If the ratio of building area in a parcel to the total building area is greater than or "
            "equal to the specified threshold, the building is assigned to the parcel."), type=float, default = 0.75, required=False)
    parser.add_argument("--threshold_low", help =(
            "A decimal value between 0 and 1 representing the building-parcel overlap threshold. "
            "If the ratio of building area in a parcel to the total building area is less than or "
            "equal to the specified threshold, the overlap is considered insignificant and will "
            "be overlooked."),type=float, default = 0.15, required=False)
    parser.add_argument("--buildings_w_parcels_path", help ='buildings_with_parcels_path', type=str, required=False, default = './buildings_processed') # to save shp
    parser.add_argument("--parcel_images_directory", help ='parcel_images_directory', type=str, required=False, default = './parcels_test/') 
    parser.add_argument("--buildings_images_directory", help ='buildings_images_directory', type=str, required=False, default = './buildings_test/') 
    parser.add_argument("--number_of_images", help ='number_of_images', type=int, default = 10, required=False)
    
    # create dataset_specs folder
    try:
        os.mkdir('./dataset_specs')
    except:
        pass
    
    # retrieve arguments
    args = parser.parse_args()
    buildings_path = args.buildings_path
    parcels_path = args.parcels_path
    threshold_high = args.threshold_high
    threshold_low = args.threshold_low
    buildings_w_parcels_path = args.buildings_w_parcels_path
    parcel_images_directory = args.parcel_images_directory
    buildings_images_directory = args.buildings_images_directory
    number_of_images = args.number_of_images
    blocks_path = args.blocks_path
    split_buildings = args.split_buildings
    load_split_buildings = args.load_split_buildings
    
    # read dataframes
    df_parcels = gpd.read_file(parcels_path).to_crs(epsg=3857)
    df_buildings = gpd.read_file(buildings_path).to_crs(epsg=3857)
    if blocks_path is not None:
        df_blocks = gpd.read_file(blocks_path).to_crs(epsg=3857)
    
    # set color to parcels
    df_parcels['color'] = [ random_hex_color() for i in range(len(df_parcels)) ]
    
    # remove duplicates from buildings and parcels
    print("Removing duplicates buildings...")
    df_buildings = remove_duplicates(df_buildings)
    print("Removing duplicates parcels...")
    df_parcels = remove_duplicates(df_parcels)
    if blocks_path is not None:
        print("Removing duplicates blocks...")
        df_blocks = remove_duplicates(df_blocks)
        # Spatial join to restrict parcels to those within blocks
        df_parcels = gpd.sjoin(df_parcels, df_blocks, op='within')
        try:
            df_parcels.drop(columns = ['index_left'], inplace=True)
        except:
            pass
        try:
            df_parcels.drop(columns = ['index_right'], inplace=True)
        except:
            pass
        try:
            df_blocks.drop(columns = ['index_left'], inplace=True)
        except:
            pass
        try:
            df_blocks.drop(columns = ['index_right'], inplace=True)
        except:
            pass
        # Reset the index of the resulting DataFrame
        df_parcels.reset_index(inplace=True, drop=True)
        df_buildings.reset_index(inplace=True, drop=True)
        df_blocks.reset_index(inplace=True, drop=True)
        
    # add building id and parcel id
    df_buildings['building_id'] = [i for i in range(len(df_buildings))]
    df_parcels['parcel_id'] = [i for i in range(len(df_parcels))]
    
    # Perform a spatial join between buildings and parcels
    buildings_with_parcel_info = df_buildings.sjoin(df_parcels, how="inner")
    
    # Count the number of times each building appears (number of intersecting parcels)
    building_counts = buildings_with_parcel_info.groupby("building_id").size()
    #Find buildings that belong to more than one parcel
    buildings_with_multiple_parcels = building_counts[building_counts > 1].index.tolist()
   
    # split buildings based on the provided thresholds
    if split_buildings:
        buildings_with_parcel_info_new = split_building(buildings_with_multiple_parcels, buildings_with_parcel_info, df_buildings, df_parcels, 
                   threshold_high = threshold_high, threshold_low = threshold_low, buildings_clean_path = buildings_w_parcels_path)
    elif load_split_buildings:    
        buildings_with_parcel_info_new = gpd.read_file(buildings_w_parcels_path).to_crs(epsg=3857)
    else:
        buildings_with_parcel_info_new = buildings_with_parcel_info.copy()
        
    ################### DATASET SPECS ###################
    generate_dataset_specs(buildings_with_parcel_info, buildings_with_parcel_info_new, df_parcels)
    
    ####### ASSIGN COLORS on buildings based on parcels
    print('Assigning colors...')
    for index, row in tqdm(buildings_with_parcel_info_new.iterrows()):
        parcel_id = row['parcel_id']
        buildings_with_parcel_info_new.loc[index,'color'] = df_parcels[df_parcels['parcel_id']==parcel_id].color.values[0]
    # Save final shp with colors
    buildings_with_parcel_info_new.to_file(buildings_w_parcels_path, driver='ESRI Shapefile')
    
    ################ IMAGE DATASETS GENERATION ################ 
    # Create directories to save images names
    try:
        os.mkdir(parcel_images_directory)
    except:
        pass
    try:
        os.mkdir(buildings_images_directory)
    except:
        pass
    
    # shuffle building indices - image index will correspond to the parcel index in the dataframe
    indices_to_print = [i for i in range(buildings_with_parcel_info_new.shape[0])]
    random.shuffle(indices_to_print)
    count = 0 
    print('Generating images...')
    for i in indices_to_print:
        try:
            if count % 1000 == 0:
                print(count+' images completed out of '+str(number_of_images))
            subset_features = subset(df_parcels, buildings_with_parcel_info_new, i, 200)
            map_maker(subset_features[0], subset_features[1], subset_features[2], i, 18, 'parcels', img_folder=parcel_images_directory)
            map_maker(subset_features[0], subset_features[1], subset_features[2], i, 18, 'buildings', img_folder=buildings_images_directory)
            count = count + 1
            if count == number_of_images:
                break
        except:
            print("Error at index: ",i)
        
    print('Done!')
        
    
