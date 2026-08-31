'use client';

import { useEffect, useRef } from 'react';
import Highcharts from 'highcharts';
import HighchartsReact from 'highcharts-react-official';
import HighchartsMore from 'highcharts/highcharts-more';
import HighchartsXRange from 'highcharts/modules/xrange';

// Initialize Highcharts modules
if (typeof Highcharts === 'object') {
  HighchartsMore(Highcharts);
  HighchartsXRange(Highcharts);
}

interface TimeWindow {
  start: string;
  end: string;
  label: string;
}

interface TimeWindowsTimelineProps {
  timeWindows: string[];
}

export default function TimeWindowsTimeline({ timeWindows }: TimeWindowsTimelineProps) {
  const chartRef = useRef<HighchartsReact.RefObject>(null);

  // Parse time windows from strings like "2020-01-01 to 2021-01-01"
  const parsedWindows: TimeWindow[] = timeWindows.map((tw, idx) => {
    const parts = tw.split(' to ');
    return {
      start: parts[0]?.trim() || '',
      end: parts[1]?.trim() || parts[0]?.trim() || '',
      label: `T${idx + 1}`,
    };
  });

  const options: Highcharts.Options = {
    chart: {
      type: 'xrange',
      backgroundColor: '#0b1120',
      height: 220,
    },
    title: {
      text: '',
    },
    xAxis: {
      type: 'datetime',
      labels: {
        style: { color: '#e5e7eb', fontSize: '10px' },
        format: '{value:%Y-%m-%d}',
      },
      gridLineColor: '#1f2933',
      lineColor: '#6b7280',
      tickColor: '#6b7280',
    },
    yAxis: {
      title: { text: '' },
      categories: parsedWindows.map((_, i) => `T${i + 1}`),
      labels: { style: { color: '#e5e7eb' } },
      gridLineColor: '#1f2933',
      reversed: true,
    },
    legend: {
      enabled: false,
    },
    plotOptions: {
      xrange: {
        borderRadius: 4,
        borderColor: '#0ea5e9',
        borderWidth: 1,
      },
    },
    series: [
      {
        type: 'xrange',
        name: 'Time Windows',
        data: parsedWindows.map((tw, i) => ({
          x: new Date(tw.start).getTime(),
          x2: new Date(tw.end).getTime(),
          y: i,
          color: 'rgba(56, 189, 248, 0.6)',
          name: `${tw.start} → ${tw.end}`,
        })),
      },
    ],
    tooltip: {
      backgroundColor: '#1f2937',
      borderColor: '#374151',
      style: { color: '#e5e7eb' },
      formatter: function () {
        const point = this.point as any;
        const start = Highcharts.dateFormat('%Y-%m-%d', point.x);
        const end = Highcharts.dateFormat('%Y-%m-%d', point.x2);
        const days = Math.round((point.x2 - point.x) / (1000 * 60 * 60 * 24));
        return `<b>${point.name}</b><br/>
                Start: ${start}<br/>
                End: ${end}<br/>
                Duration: ${days} days`;
      },
    },
    credits: {
      enabled: false,
    },
  };

  return (
    <div className="w-full">
      <HighchartsReact highcharts={Highcharts} options={options} ref={chartRef} immutable={true} />
      <p className="text-xs text-gray-400 mt-2">
        <span className="font-semibold">Time Windows:</span> {timeWindows.length}
      </p>
    </div>
  );
}
