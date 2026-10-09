# floatCSEP static dashboard

A static web front end for an experiment. floatCSEP writes the experiment once as plain
files (`floatcsep export`), and the pages read those files in the browser. No server-side
code is needed to view it: any static web server works, including `floatcsep serve`,
`python -m http.server`, GitHub Pages or a Zenodo record.

## Commands

```bash
floatcsep export <config.yml>            # writes <run_dir>/dashboard/
floatcsep export <config.yml> -o DIR     # custom output folder
floatcsep serve  <config.yml>            # exports if needed, serves on http://localhost:8765 and opens the browser
floatcsep serve  <config.yml> -p 8080 --no-browser
```

Re-run `export` after a new `floatcsep run` to refresh the data.

## Folder layout

```
dashboard/
  index.html, css/, js/, vendor/, fonts/, logos/   front end (copied from floatcsep/postprocess/web/static)
  manifest.json          experiment, region, magnitudes, time_windows, models, tests + file references
  grid.json              region cell origins, cell size, outline (GeoJSON)
  catalog.json           main catalog as column arrays (id, t [ms], lat, lon, depth, mag)
  forecasts/m<i>_w<j>.json   one file per model and window (see below)
  results/index.json     one record per (window, test, model): observed statistic, quantile, status,
                         distribution percentiles or Poisson parameters or confidence interval
  results/w*_t*_m*.json  full test distributions, loaded on demand
  figures/               pyCSEP figures copied from the results folder
```

### Forecast files

`rates` (cells x magnitude bins) are turned into cumulative rates per threshold
(M >= m_k), then log10 and quantized to uint16: `0` means no rate, `1..65535` map linearly
onto `[vmin, vmax]`. `data` is the base64 of that array (row-major, cells x thresholds);
`totals` is the expected number of events per threshold; `grid` is `"region"` when the
forecast shares the experiment grid, otherwise its own `{dh, origins}`.

## Front end

Plain ES modules, no build step. `js/app.js` loads the manifest, routes `#/experiment`,
`#/catalogs`, `#/forecasts`, `#/results` to `js/pages/*.js`, and handles the theme.
Maps are Leaflet (`vendor/leaflet`), charts are Apache ECharts (`vendor/echarts`), the
forecast raster is a canvas layer (`js/gridlayer.js`). Colours and spacing are CSS custom
properties in `css/app.css` (light and dark).

Basemap tiles come from CARTO, Esri and OpenStreetMap, so maps need internet access;
everything else works offline.
