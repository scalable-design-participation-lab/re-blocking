"""
Million Neighborhoods `Re-Blocking`
A library for generating and validating parcel data for informal settlements
"""

class Parcels:
    """🐕"""
    name: str
    """The name of our dog."""
    friends: list["Dog"]
    """The friends of our dog."""

    def __init__(self, name: str):
        """Make a Dog without any friends (yet)."""
        self.name = name
        self.friends = []

    def bark(self, loud: bool = True):
        """*woof*"""

Read dataframes
Set color to parcels
Remove duplicates from buildings and parcels
Spatial join to restrict parcels to those within blocks
Reset the index of the resulting DataFrame
Add building id and parcel id
Perform a spatial join between buildings and parcels
Count the number of times each building appears (number of intersecting parcels)
Find buildings that belong to more than one parcel
Split buildings based on the provided thresholds
DATASET SPECS
ASSIGN COLORS on buildings based on parcels
Save final shp with colors
IMAGE DATASETS GENERATION
Create directories to save images names
Shuffle building indices - image index will correspond to the parcel index in the data frame
Generate Images


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