import { VIRIDIS_COLORS } from '@/lib/palettes';
import DualRangeSlider from '@/components/ui/DualRangeSlider';

interface ColorbarLegendProps {
  vmin: number;
  vmax: number;
  dataMin?: number; // The absolute min of the data
  dataMax?: number; // The absolute max of the data
  title?: string;
  onRangeChange?: (range: [number, number]) => void;
}

// Using Viridis to match the map
const GRADIENT_COLORS = VIRIDIS_COLORS;

export default function ColorbarLegend({
  vmin,
  vmax,
  dataMin,
  dataMax,
  title = 'log10 λ',
  onRangeChange
}: ColorbarLegendProps) {

  // Use data range if provided, otherwise fallback to current vmin/vmax with some padding logic if needed
  // In our case, app provides explicit data range
  const absMin = dataMin !== undefined ? dataMin : vmin - 1;
  const absMax = dataMax !== undefined ? dataMax : vmax + 1;

  // Ensure effective range covers current view
  const minLimit = Math.min(absMin, vmin);
  const maxLimit = Math.max(absMax, vmax);

  return (
    <div className="bg-surface p-4 rounded-lg border border-border space-y-4">
      <div className="flex items-center gap-3">
        <span className="text-xs text-gray-400 font-mono w-16 text-right">{vmin.toFixed(2)}</span>
        <div className="flex-1 h-8 rounded border border-border relative overflow-hidden">
          <div
            className="absolute inset-0 w-full h-full"
            style={{
              background: `linear-gradient(to right, ${GRADIENT_COLORS.join(', ')})`,
            }}
          />
        </div>
        <span className="text-xs text-gray-400 font-mono w-16">{vmax.toFixed(2)}</span>
      </div>

      {onRangeChange && (
        <div className="px-16 + 3"> {/* Padding to align with the gradient bar roughly */}
          <DualRangeSlider
            min={minLimit}
            max={maxLimit}
            step={0.1}
            value={[vmin, vmax]}
            onValueChange={onRangeChange}
          />
          <div className="flex justify-between text-[10px] text-gray-500 mt-1">
            <span>Data Min: {minLimit.toFixed(2)}</span>
            <span>Data Max: {maxLimit.toFixed(2)}</span>
          </div>
        </div>
      )}

      <div className="text-center">
        <span className="text-xs text-gray-400">{title}</span>
      </div>
    </div>
  );
}
