import os
import glob
import rasterio
import pandas as pd
from src.config import get_station_meta

meta = get_station_meta()
files = glob.glob('data/TROPOMI_Optimized_Cloud/*20200226*.tif')

print(f"Loaded {len(meta)} stations. Found {len(files)} files.")

results = []
for f in files:
    with rasterio.open(f) as src:
        # Sample points
        coords = [(row.longitude, row.latitude) for _, row in meta.iterrows()]
        values = list(src.sample(coords))
        
        for i, (station, _) in enumerate(meta.iterrows()):
            val = values[i][0]
            if val != src.nodata and val != 0.0 and val > -9999: # often 0 is nodata if not set, or maybe np.isnan
                import numpy as np
                if not np.isnan(val):
                    results.append((station, os.path.basename(f), val))

df_res = pd.DataFrame(results, columns=['station', 'file', 'val'])
print(df_res.head(20))
