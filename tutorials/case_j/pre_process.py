import os
import shutil

import csep
import pandas as pd
import numpy as np

start_time = '2025-04-04'
end_time = '2025-05-11'

# Parse forecast
forecast = pd.read_csv('result.csv')

# realization_id,magnitude,depth,latitude,longitude,time
# lon,lat,m,time,depth,catalog_id,event_id

os.makedirs('models/etas/forecasts/', exist_ok=True)
forecast_data = pd.DataFrame({'lon': forecast.longitude,
                              'lat': forecast.latitude,
                              'mag': forecast.magnitude,
                              'time': [i.replace(" ", 'T') for i in forecast.time],
                              'depth': forecast.depth/1000.,
                              'cat_id': forecast.realization_id,
                              'event_id': np.arange(len(forecast.longitude))})


forecast_data.to_csv(f'models/etas/forecasts/etas_{start_time}_{end_time}.csv', index=False)

shutil.copy('result.csv', f'models/etas/forecasts/etas2_{start_time}_{end_time}.csv')