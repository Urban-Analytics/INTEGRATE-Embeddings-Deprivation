import os
from pathlib import Path

data_dir =   os.path.join("..", "data", "processed")
lsoas_file = os.path.join("..", "data", "raw", "spatial", "LSOAs_2021", "LSOA_2021_EW_BSC_V4.shp")
imd_file =   os.path.join("..", "data", "raw", "imd", "File_2_-_IoD2025_Domains_of_Deprivation.xlsx")

h5_filename = os.path.join("..", "data", "processed", "street_data.h5")

outputs_dir = os.path.join(data_dir, "..", "outputs")