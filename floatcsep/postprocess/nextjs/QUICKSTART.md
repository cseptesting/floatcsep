# Quick Start Guide: Next.js Dashboard

## Installation

### 1. Install Node.js 20+

```bash
# Using nvm (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 20
nvm use 20
node --version  # Should show v20.x.x
```

### 2. Verify floatCSEP Installation

```bash
# Make sure floatCSEP is installed
python -c "import floatcsep; print(floatcsep.__version__)"
```

## Usage

### Launch the Dashboard

From your experiment directory:

```bash
# Launch Next.js dashboard (recommended)
floatcsep view config.yaml --ui nextjs

# Or use the default Panel dashboard
floatcsep view config.yaml --ui panel
# Or simply: floatcsep view config.yaml
```

**What happens:**
1. Python loads your experiment configuration
2. Builds a manifest with all metadata, models, tests, catalogs
3. Installs Next.js dependencies (first run only - takes ~2 minutes)
4. Starts the Next.js development server
5. Opens your browser automatically

**First Run:**
- Takes 2-3 minutes to install npm packages
- Subsequent runs start in ~5 seconds

### Dashboard Features

#### Experiment Tab ✅
- **Metadata**: Authors, DOI, versions, license
- **Configuration**: Temporal/spatial settings, models, tests
- **Visualizations**:
  - Interactive region map (Leaflet)
  - Time windows timeline (Highcharts)

#### Catalogs Tab ✅
- **Spatial Map**: Earthquake events with magnitude-based sizing
  - Blue circles: Input catalog (t < start date)
  - Red circles: Test catalog (t ≥ start date)
  - Click for event details
- **Magnitude-Time Plot**: Interactive scatter plot
  - Time windows shown as light blue bands
  - Zoom and pan enabled

#### Forecasts Tab 🚧
- Coming soon: Gridded forecast visualization with color controls

#### Results Tab 🚧
- Coming soon: Evaluation results with dynamic PNG loading

## Keyboard Shortcuts

- **Ctrl+C**: Stop the server
- **Browser refresh**: Reload data

## Troubleshooting

### "unrecognized arguments: --ui nextjs"

**Solution**: Make sure you've updated floatCSEP to include the CLI changes:
```bash
cd /path/to/floatcsep
pip install -e .
```

### "Node.js 20+ required"

**Solution**: Install Node.js 20 using nvm (see Installation above)

### "Cannot find module 'next'"

**Solution**: Dependencies not installed. The server will install them automatically on first run, or manually:
```bash
cd floatcsep/postprocess/nextjs
npm install
```

### Server won't start

**Solution**: Check if port is already in use:
```bash
# Find process using port 3000
lsof -i :3000
# Kill it if needed
kill -9 <PID>
```

### Manifest not found

**Error**: `Manifest path not configured`

**Solution**: This is handled automatically by the Python launcher. If you see this error, ensure you're launching via:
```bash
floatcsep view config.yaml --ui nextjs
```

Not via:
```bash
npm run dev  # This requires manual environment setup
```

## Development Mode

For dashboard development (hot reload enabled):

```bash
# 1. First, generate a manifest manually
cd /path/to/your/experiment
floatcsep view config.yaml --ui panel
# This creates results directory with manifest data

# 2. Set environment variables
export MANIFEST_PATH="/path/to/floatcsep/postprocess/nextjs/.cache/manifest.json"
export APP_ROOT="/path/to/your/experiment/results"

# 3. Start dev server
cd floatcsep/postprocess/nextjs
npm run dev
```

Then open http://localhost:3000

## Performance Tips

- **First Load**: ~2-3 seconds (loads manifest + catalog)
- **Tab Switching**: Instant (data cached via SWR)
- **Data Updates**: Auto-cached for 1-5 minutes
- **Browser Cache**: Assets cached for 1 hour

## What's Working

✅ **Phases 1-3 Complete:**
- CLI integration with `--ui` flag
- Experiment metadata with maps and timelines
- Catalog visualization (spatial + temporal)
- Modern dark theme UI
- Python-Next.js integration

🚧 **Coming Soon (Phases 4-5):**
- Forecasts tab with gridded data
- Results tab with PNG viewer

## Support

If you encounter issues:

1. Check the [README.md](README.md) for detailed documentation
2. Verify Node.js version: `node --version` (should be 20+)
3. Check server logs in the terminal
4. Try the Panel UI as fallback: `floatcsep view config.yaml --ui panel`

## Example Workflow

```bash
# 1. Run your experiment
floatcsep run config.yaml

# 2. View results in Next.js dashboard
floatcsep view config.yaml --ui nextjs

# 3. Browser opens automatically at http://localhost:XXXX
#    - Navigate tabs using top navigation
#    - Explore experiment metadata
#    - Visualize catalog events
#    - (Soon) View forecasts and results

# 4. Stop server with Ctrl+C
```

## Comparison: Panel vs Next.js

| Feature | Panel | Next.js |
|---------|-------|---------|
| **UI Framework** | Bokeh | React + Tailwind |
| **Maps** | Bokeh Tiles | Leaflet |
| **Charts** | Bokeh | Highcharts |
| **Startup** | Fast (~2s) | First: ~3min, Then: ~5s |
| **Browser** | Any | Modern (Chrome 90+) |
| **Customization** | Python | TypeScript/React |
| **Performance** | Good | Excellent (with caching) |
| **Mobile** | Limited | Responsive |

**Recommendation**: Use Next.js for modern UI and better interactivity. Use Panel for quick results viewing without Node.js setup.

## Next Steps

After getting familiar with the dashboard:
- Explore different experiments
- Compare results across time windows
- (Soon) Interact with forecast visualizations
- (Soon) Export figures from results tab

Happy exploring! 🚀
