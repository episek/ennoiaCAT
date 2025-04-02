'use client';
import { useState, useEffect } from 'react';
import styles from './page.module.css';
import LoadingSpinner from './components/LoadingSpinner';
import SpectrumChart from './components/SpectrumChart';
import SpectrumSummary from './components/SpectrumSummary';
import { fetchSpectrumData } from './services/spectrumApi';
import { ApiResponse } from './types/spectrum';
import imageSrc from './components/ennoia.jpg';

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<ApiResponse | null>(null);
  useEffect(() => {
    console.log("DEBUG");
    console.log(data);
  }, [data]);
  const handleAnalyze = async () => {
    setLoading(true);
    try {
      const spectrumData = await fetchSpectrumData();
      setData(spectrumData);
    } catch (error) {
      console.error('Failed to fetch spectrum data:', error);
    } finally {
      setLoading(false);
    }
  };

  return (

<div className={styles.page}>
  <div className={`${styles.gradientBg} ${(loading || data) ? styles.active : ''}`} />
  <main className={`${styles.main} ${(loading || data) ? styles.analyzing : ''}`}>
    {!loading && !data && (
      <>
        <h1 className={styles.title}>AI assisted Interference Detection</h1> {/* Title added here */}
         {/* Adding an image */}
         <img 
          src= {imageSrc.src}
          alt="Ennoia Technologies" 
          //className={styles.imageClass} // Optional: add a custom class for styling
        />
        <button 
          className={styles.analyzeButton} 
          onClick={handleAnalyze}
        >
          Analyze
        </button>
      </>
    )}
 

        {loading && (
          <div className={styles.spinnerWrapper}>
            <LoadingSpinner />
          </div>
        )}
        
        {data && (
          <>
          {data && data.frequency_report?.frequency_ranges?.length > 0 && (
          <>
            <div className={styles.chartContainer}>
              <SpectrumChart 
                frequencies={data.frequency}
                signalStrengths={data.signal_strengths}
              />
            </div>
            <SpectrumSummary data={data.frequency_report.frequency_ranges} />
          </>
        )}
        </>
        )}
      </main>
    </div>
  );
}
