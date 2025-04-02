'use client';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface Props {
  frequencies: number[];
  signalStrengths: number[];
}

export default function SpectrumChart({ frequencies, signalStrengths }: Props) {
  const chartData = {
    labels: frequencies.map(f => `${f.toFixed(1)} MHz`),
    datasets: [
      {
        label: 'Signal Strength',
        data: signalStrengths,
        borderColor: '#3498db',
        backgroundColor: 'rgba(52, 152, 219, 0.2)',
        borderWidth: 2,
        pointBackgroundColor: '#3498db',
        pointRadius: 1, // Make points smaller due to large dataset
        tension: 0.1,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: '#fff'
        }
      },
      title: {
        display: true,
        text: 'Frequency Spectrum Analysis',
        color: '#fff'
      },
    },
    scales: {
      y: {
        title: {
          display: true,
          text: 'Signal Strength (dB)',
          color: '#fff'
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        ticks: {
          color: '#fff'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Frequency (MHz)',
          color: '#fff'
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        ticks: {
          color: '#fff',
          maxRotation: 45,
          autoSkip: true,
          maxTicksLimit: 20
        }
      },
    },
  };

  return <Line data={chartData} options={options} />;
} 