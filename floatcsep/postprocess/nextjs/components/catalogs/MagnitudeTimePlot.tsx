'use client';

import { useRef, useEffect } from 'react';
import { useTheme } from 'next-themes';
import Highcharts from 'highcharts';
import HighchartsReact from 'highcharts-react-official';
import HC_exporting from 'highcharts/modules/exporting';
import HC_exportData from 'highcharts/modules/export-data';
import HC_offlineExporting from 'highcharts/modules/offline-exporting';
import { CatalogEvent } from '@/lib/types';

// Initialize exporting modules
if (typeof Highcharts === 'object') {
  HC_exporting(Highcharts);
  HC_offlineExporting(Highcharts);
  HC_exportData(Highcharts);
}

interface MagnitudeTimePlotProps {
  events: CatalogEvent[];
  timeWindows: string[];
  startDate?: string;
}

export default function MagnitudeTimePlot({
  events,
  timeWindows,
  startDate,
}: MagnitudeTimePlotProps) {
  const chartRef = useRef<HighchartsReact.RefObject>(null);
  const { theme } = useTheme();

  useEffect(() => {
    // Ensure modules are initialized on client side
    if (typeof Highcharts === 'object') {
      HC_exporting(Highcharts);
      HC_offlineExporting(Highcharts);
      HC_exportData(Highcharts);
    }
  }, []);

  // Categorize events
  const inputEvents = events.filter((e) =>
    startDate ? new Date(e.time) < new Date(startDate) : false
  );
  const testEvents = events.filter((e) =>
    startDate ? new Date(e.time) >= new Date(startDate) : true
  );

  // Parse time windows for plot bands
  const plotBands = timeWindows.map((tw) => {
    const parts = tw.split(' to ');
    return {
      from: new Date(parts[0]?.trim() || '').getTime(),
      to: new Date(parts[1]?.trim() || parts[0]?.trim() || '').getTime(),
      color: theme === 'dark' ? 'rgba(56, 189, 248, 0.1)' : 'rgba(56, 189, 248, 0.2)',
    };
  });

  const isDark = theme === 'dark';
  const textColor = isDark ? '#e5e7eb' : '#1f2937';
  const gridColor = isDark ? '#1f2933' : '#e5e7eb';
  const backgroundColor = isDark ? '#0b1120' : '#ffffff';

  const options = {
    chart: {
      type: 'scatter',
      backgroundColor: backgroundColor,
      zoomType: 'x',
      height: 500,
      style: {
        fontFamily: 'var(--font-noto-sans)',
      },
    },
    title: {
      text: '',
    },
    xAxis: {
      type: 'datetime',
      title: {
        text: 'Time',
        style: { color: textColor },
      },
      labels: {
        style: { color: textColor, fontSize: '12px' },
        format: '{value:%Y-%m-%d}',
      },
      gridLineColor: gridColor,
      lineColor: isDark ? '#6b7280' : '#cbd5e1',
      tickColor: isDark ? '#6b7280' : '#cbd5e1',
      plotBands: plotBands,
    },
    yAxis: {
      title: {
        text: 'Magnitude',
        style: { color: textColor },
      },
      labels: {
        style: { color: textColor },
      },
      gridLineColor: gridColor,
    },
    legend: {
      itemStyle: { color: textColor },
    },
    plotOptions: {
      scatter: {
        marker: {
          radius: 4,
          states: {
            hover: {
              enabled: true,
              lineColor: textColor,
            },
          },
        },
        states: {
          hover: {
            marker: {
              enabled: false,
            },
          },
        },
      },
    },
    series: [
      {
        name: 'Input Catalog',
        type: 'scatter',
        data: inputEvents.map((e) => ({
          x: new Date(e.time).getTime(),
          y: e.magnitude,
          name: e.event_id.replace(/^b'|'$/g, ''),
        })),
        color: '#38bdf8',
        marker: {
          fillOpacity: 0.6,
        },
      },
      {
        name: 'Test Catalog',
        type: 'scatter',
        data: testEvents.map((e) => ({
          x: new Date(e.time).getTime(),
          y: e.magnitude,
          name: e.event_id.replace(/^b'|'$/g, ''),
        })),
        color: '#ef4444',
        marker: {
          fillOpacity: 0.8,
        },
      },
    ],
    tooltip: {
      backgroundColor: isDark ? '#1f2937' : '#ffffff',
      borderColor: isDark ? '#374151' : '#e5e7eb',
      style: { color: textColor },
      formatter: function (this: any) {
        const point = this.point;
        return `<b>${point.name}</b><br/>
                Time: ${Highcharts.dateFormat('%Y-%m-%d %H:%M:%S', this.x)}<br/>
                Magnitude: ${this.y?.toFixed(2)}`;
      },
    },
    exporting: {
      enabled: true,
      buttons: {
        contextButton: {
          symbolStroke: textColor,
          theme: {
            fill: isDark ? '#1f2937' : '#f3f4f6',
            stroke: gridColor,
          }
        }
      }
    },
    navigation: {
      buttonOptions: {
        theme: {
          stroke: textColor,
          style: {
            color: textColor
          }
        }
      }
    },
    credits: {
      enabled: false,
    },
  };

  return (
    <div className="w-full">
      <HighchartsReact highcharts={Highcharts} options={options} ref={chartRef} immutable={false} />
    </div>
  );
}
