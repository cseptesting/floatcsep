# floatCSEP Next.js Dashboard

A modern, interactive web dashboard for floatCSEP earthquake forecasting experiments built with Next.js, React, Leaflet, and Highcharts.

## Features

### ✅ Implemented (Phases 1-7)

- **Experiment Tab**: Interactive metadata accordion with Leaflet region map and Highcharts time windows timeline
- **Catalogs Tab**: Spatial event map with magnitude-based sizing and magnitude-time scatter plot with time window overlays
- **Forecasts Tab**: Interactive gridded forecast visualization with Magma color palette, adjustable color range controls, model/time window selectors
- **Results Tab**: Dynamic PNG loading with cascading selectors (time window → test → model)
- **Modern UI**: Dark theme with Tailwind CSS, responsive design, loading states, error boundaries
- **Python Integration**: Seamless integration with floatCSEP via subprocess API calls
- **Data Caching**: Multi-level caching (SWR client-side + in-memory server-side)
- **Accessibility**: ARIA labels, keyboard navigation, skip links, focus management

## Installation

### Prerequisites

- **Node.js 20+** (required for Next.js 14)
- **Python 3.9-3.12** with floatCSEP installed
- **npm** (comes with Node.js)

### Setup

1. **Install Node.js 20** (if not already installed):
   ```bash
   # Using nvm (recommended)
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
   source ~/.bashrc
   nvm install 20
   nvm use 20
   ```

2. **Install dependencies** (automatic on first run):
   ```bash
   cd floatcsep/postprocess/nextjs
   npm install
   ```

## Usage

### Launch Dashboard

From your experiment directory:

```bash
# Launch Next.js dashboard
floatcsep view config.yaml --ui nextjs

# Or use default Panel dashboard
floatcsep view config.yaml --ui panel  # or just: floatcsep view config.yaml
```

The Next.js server will:
1. Build experiment manifest from your config
2. Install dependencies (first run only)
3. Start development server
4. Auto-open browser at `http://localhost:PORT`

### Development Mode

For dashboard development:

```bash
cd floatcsep/postprocess/nextjs

# Set environment variables
export MANIFEST_PATH="/path/to/.cache/manifest.json"
export APP_ROOT="/path/to/experiment"

# Start dev server with hot reload
npm run dev
```

### Production Build

```bash
npm run build
npm run start
```

## Architecture

### Data Flow

```
floatcsep view config.yaml --ui nextjs
    ↓
Experiment.from_yml() → build_manifest()
    ↓
manifest.json saved to .cache/
    ↓
Next.js server starts (npm run dev)
    ↓
Browser opens → React loads → fetch /api/manifest
    ↓
User interacts → API calls Python subprocess → returns JSON
```

### Directory Structure

```
nextjs/
├── app/                    # Next.js App Router
│   ├── layout.tsx          # Root layout
│   ├── page.tsx            # Home (redirects to /experiment)
│   ├── experiment/         # Experiment tab
│   ├── catalogs/           # Catalogs tab
│   ├── forecasts/          # Forecasts tab (pending)
│   ├── results/            # Results tab (pending)
│   └── api/                # API routes
│       ├── manifest/       # GET manifest.json
│       ├── catalog/data/   # GET catalog events (via Python)
│       └── forecasts/data/ # GET forecast grid (pending)
├── components/
│   ├── layout/             # Header, Footer, Navigation
│   ├── experiment/         # RegionMap, TimeWindowsTimeline
│   ├── catalogs/           # SpatialMap, MagnitudeTimePlot
│   └── ui/                 # shadcn/ui components
├── lib/
│   ├── contexts/           # React Context providers
│   ├── types.ts            # TypeScript interfaces
│   ├── utils.ts            # Utility functions
│   └── api-client.ts       # SWR data fetching hooks
├── manifest_api.py         # Python subprocess handlers
├── server.py               # Python launcher
└── __init__.py             # Python module exports
```

### API Endpoints

| Endpoint | Method | Purpose | Caching | Python Subprocess |
|----------|--------|---------|---------|-------------------|
| `/api/manifest` | GET | Get experiment manifest | 1 hour | No (reads JSON) |
| `/api/catalog/data` | GET | Get catalog events | SWR 60s | Yes (`load_catalog_data`) |
| `/api/forecasts/data` | GET | Get forecast grid | In-memory + SWR 5min | Yes (`load_forecast_data`) |
| `/api/results/[...path]` | GET | Stream result PNG | 1 hour | No (direct file read) |

### Key Technologies

- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first styling
- **Leaflet**: Interactive maps with dark basemaps
- **Highcharts**: Charts (scatter, timeline, x-range)
- **SWR**: Data fetching with caching
- **Radix UI**: Accessible components (Accordion)

## Components

### Experiment Tab

- **Metadata Accordion**: Authors, DOI, versions, configurations
- **Region Map**: Leaflet map with region bounding box overlay
- **Time Windows Timeline**: Highcharts x-range chart showing forecast intervals

### Catalogs Tab

- **Spatial Map**: Leaflet map with earthquake events as circles
  - Marker size based on magnitude (exponential scaling)
  - Color-coded: Input (blue, t < start) vs Test (red, t ≥ start)
  - Hover tooltips with event details
- **Magnitude-Time Plot**: Highcharts scatter plot
  - X-axis: datetime, Y-axis: magnitude
  - Time windows shown as plot bands
  - Interactive zoom/pan

### Forecasts Tab

- **Forecast Map**: Leaflet map with gridded forecast overlay
  - Color-coded cells using Magma palette (log10 scale)
  - Interactive hover tooltips with rate values
  - Dynamic updates based on model/time window selection
- **Color Controls**: Adjustable min/max range for color mapping
  - Reset to data range
  - Live updates on map
- **Model Selector**: Dropdown with model metadata (DOI, Zenodo)
- **Time Window Selector**: Choose forecast period
- **Statistics Panel**: Cell count, rate range

### Results Tab

- **Cascading Selectors**: Time Window → Test → Model
- **Dynamic Image Loading**: PNG results with error handling
- **Test Metadata Panel**: Test type, function, percentile
- **Model Metadata Panel**: DOI links, Zenodo IDs

## Configuration

### Environment Variables

- `MANIFEST_PATH`: Path to manifest.json file
- `APP_ROOT`: Experiment root directory
- `PORT`: Server port (default: auto-assigned)
- `HOSTNAME`: Server hostname (default: localhost)

### Tailwind Theme

Custom colors defined in `tailwind.config.ts`:

```typescript
colors: {
  background: '#020617',   // Page background
  foreground: '#e5e7eb',   // Text color
  surface: '#0b1120',      // Card/surface background
  border: '#1f2933',       // Border color
  primary: '#14b8a6',      // Teal (floatCSEP brand)
  secondary: '#f59e0b',    // Amber
  input: '#38bdf8',        // Light blue (input catalog)
  test: '#ef4444',         // Red (test catalog)
}
```

## Python Integration

### Manifest Serialization

The `server.py` module converts the Python `Manifest` dataclass to JSON:

```python
from floatcsep.postprocess.nextjs import run_nextjs_app

run_nextjs_app(experiment, port=0, show=True, mode='dev')
```

### Subprocess Handlers

`manifest_api.py` provides CLI interface for data processing:

```bash
# Load catalog
python manifest_api.py load_catalog "path/to/catalog.csep" "/app/root"

# Load forecast
python manifest_api.py load_forecast "path/to/forecast.dat" "/app/root" '{"bbox":[...]}' false
```

## Development

### Adding New Components

1. Create component in `components/`
2. Use `'use client'` directive for interactive components
3. Import dynamically in pages to avoid SSR issues:

```typescript
const MyMap = dynamic(() => import('@/components/MyMap'), {
  ssr: false,
  loading: () => <LoadingSkeleton />,
});
```

### Adding New API Routes

1. Create `app/api/my-endpoint/route.ts`
2. Export `GET`, `POST`, etc. handlers
3. Call Python subprocess if needed:

```typescript
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export async function GET(request: NextRequest) {
  const { stdout } = await execAsync('python manifest_api.py my_command args');
  return NextResponse.json(JSON.parse(stdout));
}
```

### Styling Guidelines

- Use Tailwind utility classes
- Follow dark theme color scheme
- Use consistent spacing (4px increments)
- Ensure text contrast (WCAG AA minimum)

## Troubleshooting

### Node.js version error

```
Error: create-next-app requires Node.js 20+
```

**Solution**: Install Node.js 20 using nvm:
```bash
nvm install 20
nvm use 20
```

### Dependencies not installed

```
Error: Cannot find module 'next'
```

**Solution**: Run `npm install` in the nextjs directory

### Manifest not found

```
Error: Manifest path not configured
```

**Solution**: Ensure `MANIFEST_PATH` environment variable is set by the Python launcher

### Python subprocess fails

```
Error: Failed to load catalog data
```

**Solution**: Check that:
1. Python is in PATH
2. floatCSEP is installed in the Python environment
3. Catalog file exists at the specified path

## Performance

- **Client-side caching**: SWR deduplicates requests (60s for manifest/catalog, 5min for forecasts)
- **Lazy loading**: Maps and charts load only when needed (dynamic imports)
- **Optimized builds**: Production builds minified and tree-shaken
- **Static assets**: Served with cache headers (1 hour)

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## License

Same as floatCSEP (see main repository)

## Contributing

This dashboard is part of the floatCSEP project. For contributions:

1. Follow existing code style
2. Test with real experiment data
3. Ensure backward compatibility with Panel dashboard
4. Update this README if adding features

## Roadmap

- [x] **Phase 1**: Foundation (Next.js setup, Python integration, CLI modification)
- [x] **Phase 2**: Experiment tab (metadata accordion, region map, timeline)
- [x] **Phase 3**: Catalogs tab (spatial map, magnitude-time plot)
- [x] **Phase 4**: Forecasts tab (gridded data overlay with color controls)
- [x] **Phase 5**: Results tab (PNG viewer with cascading selectors)
- [x] **Phase 6**: Polish (error boundaries, accessibility, ARIA labels, keyboard navigation)
- [x] **Phase 7**: Documentation (updated README, QUICKSTART)

### Future Enhancements

- [ ] Export to static site (Next.js `output: 'export'`)
- [ ] Offline support with service workers
- [ ] Advanced filtering for catalogs (magnitude, depth range)
- [ ] Forecast comparison view (side-by-side models)
- [ ] Export visualizations as PNG/SVG
- [ ] Custom color palettes for forecasts

## Credits

Built by the floatCSEP team using:
- [Next.js](https://nextjs.org/)
- [Leaflet](https://leafletjs.com/)
- [Highcharts](https://www.highcharts.com/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Radix UI](https://www.radix-ui.com/)
