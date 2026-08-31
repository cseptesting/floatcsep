export interface Manifest {
  name: string;
  start_date: string;
  end_date: string;
  authors: string | null;
  doi: string | null;
  journal: string | null;
  manuscript_doi: string | null;
  exp_time: string | null;
  floatcsep_version: string | null;
  pycsep_version: string | null;
  last_run: string | null;
  catalog_doi: string | null;
  license: string | null;
  date_range: string;
  magnitudes: number[];
  region: Region | null;
  models: Model[];
  tests: Test[];
  time_windows: string[];
  catalog: CatalogInfo;
  results_main: Record<string, string>;
  results_model: Record<string, string>;
  app_root: string;
  exp_class: string;
  n_intervals: number;
  horizon: string | null;
  offset: string | null;
  growth: string | null;
  mag_min: number | null;
  mag_max: number | null;
  mag_bin: number | null;
  depth_min: number | null;
  depth_max: number | null;
  run_mode: string | null;
  run_dir: string | null;
  config_file: string | null;
  model_config: string | null;
  test_config: string | null;
}

export interface Region {
  name: string | null;
  bbox: [number, number, number, number] | null; // [west, south, east, north]
  dh: number | null;
  origins: [number, number][] | null;
}

export interface Model {
  name: string;
  forecast_unit: string | null;
  path: string | null;
  giturl: string | null;
  git_hash: string | null;
  zenodo_id: string | null;
  authors: string | null;
  doi: string | null;
  func: string | null;
  func_kwargs: Record<string, any> | null;
  fmt: string | null;
  forecasts: Record<string, string>;
  forecast_class: string;
  forecast_paths?: string[] | null;
  is_catalog_forecast?: boolean;
}

export interface Test {
  name: string;
  func: string | null;
  func_kwargs: Record<string, any> | null;
  ref_model: string | null;
  plot_func: string | null;
  plot_args: any[] | null;
  plot_kwargs: Record<string, any> | null;
  type?: string | null;
  percentile?: number | null;
}

export interface CatalogInfo {
  path?: string;
  map?: string;
  time?: string;
}

export interface CatalogEvent {
  lon: number;
  lat: number;
  magnitude: number;
  time: string;
  event_id: string;
  category?: 'input' | 'test';
}

export interface CatalogData {
  events: CatalogEvent[];
  count: number;
  bbox: [number, number, number, number] | null;
}

export interface ForecastCell {
  lon: number;
  lat: number;
  rate: number;
}

export interface ForecastData {
  cells: ForecastCell[];
  vmin: number;
  vmax: number;
}
