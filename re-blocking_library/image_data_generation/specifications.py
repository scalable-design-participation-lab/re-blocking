"""
Million Neighborhoods `Re-Blocking`
A library for generating and validating parcel data for informal settlements
"""

class Parcels:
    """🐕"""
    parcel_path: str
    """Location of the parcel SHP file."""
    buildings_path: str
    """Location of the buildings SHP file."""
    blocks_path: str
    """Location of the block SHP file."""

    def __init__(self, parcel_path: str, buildings_path: str, blocks_path: str or None = None):
        """Create a parcel block geodataframe."""
        self.parcel_path = parcel_path
        self.buildings_path = buildings_path
        self.blocks_path = blocks_path

    # def bark(self, loud: bool = True):
    #     """*woof*"""

Functions:
Every numbered element below is a function that can be called by the Parcel object.
Helper functions are now functions that are part of the class.
1- Init:
    Create df's based on the paths provided 
    If blocks_path is provided, restrict parcels to those within blocks
    Make sure that building and parcel id is the same by joining the two dataframes
    Remove duplicates from buildings and parcels

2- Split buildings:
    Find buildings that belong to more than one parcel
    Split buildings based on the provided thresholds
    Update the building dataframe with the split buildings

3- Assign colors:
    Set color to parcels
    This will depend on whether the parcels are split or not, if they are split, the color will be assigned to the split parcels

4- Generate dataset specs:
    Count the number of times each building appears (number of intersecting parcels)
    Generates a set of specitications for the parcels and buildings (sort of what generate_dataset_specs does)

5- Generate images:
    Create directories to save buildings and parcels with buildings
    Generate Images
    
6- random_hex_color:
    Generate a random hex color
    
7- remove_duplicates:
    Remove duplicates from a dataframe
    
8- add geometries:
    Add geometries to a map plot

9- plot a map 
    calculate the centroid and max distance of the bounds
    Plot a map with parcels and buildings or just with Parcels
    save the image
    
10- split_building:
    split a building when it falls within multiple parcels based on the provided thresholds
    returns the split buildings dataframe
    
11- generate_dataset_specs:
    Generate a set of specifications for the parcels and buildings
    creates plots with histograms
    creates a summary of the data as a text file
    
    
    


## Helper
Loop through the parcels and add them to the map
Subset the data frames based on a buffer
Change figure size of the subplot
Calculate the centroid and max distance of the bounds
Calculate the centroid of the bounds
# bounds = df_parcels.total_bounds with offset to create same aspect ratio
Add the Mapbox tiles to the axis at zoom level 10 (Mapbox has 23 zoom levels)
Save the figure
Close the figure
Create an empty GeoDataFrame to hold the split buildings
Initialize a counter for tracking progress
Create a dictionary to store information about parcels to be removed
Iterate through buildings with multiple parcels
# Print progress every 1000 buildings
# Increment the counter
Select the building with the current building_id
Get the list of parcel IDs that intersect with the building
Initialize lists to store split geometries and areas
Split the building geometry into separate parts based on the parcels
Calculate the normalized areas for each parcel
Find the maximum normalized area value and its index
If the maximum area is greater than or equal to threshold_high, remove everything except the one with high area
# If the maximum area is less than threshold_high, remove all parcels below 0.15 and between 0.15 and threshold_high
Create a temporary building with split geometry and update its attributes
Append the temporary building to the split_buildings GeoDataFrame
# add new ones
save geopandas
Calculate number of buildings before/after split
Spatial join to associate each building with a parcel
Group by parcel ID and count buildings
Filter parcels with more than one building
Calculate parcel areas
Plot histogram of parcel areas
Generate descriptive statistics for the 'area' column
Spatial join to associate buildings with parcels for coverage calculation
Calculate parcel areas and coverage percentage
Generate descriptive statistics for the 'coverage_percentage' column
Save initial summary information
Save area statistics
Save coverage statistics