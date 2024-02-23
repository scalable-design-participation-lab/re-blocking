Re-blocking code as script


Image Generation Process
------------------------

1.	Load datasets (buildings, parcels, blocks)
2.	Fix projections
3.	Remove duplicates (buildings, parcels, blocks)
4.	If blocks: restrict parcels with blocks
5.	Restrict buildings with parcels
6.	Split buildings (default threshold)
    > Save split buildings 
7.	Calculate dataset specs
    > Write results in txt 
8.	Assign colors to parcels and buildings
    > Save new split buildings
9.	Generate image datasets

