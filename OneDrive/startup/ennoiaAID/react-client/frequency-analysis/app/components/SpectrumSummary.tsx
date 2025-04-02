import { SpectrumData } from '../types/spectrum';
import styles from './SpectrumSummary.module.css';

interface Props {
  data: SpectrumData[];
}

export default function SpectrumSummary({ data }: Props) {
  return (
    <div className={styles.summaryContainer}>
      <h2>Frequency Spectrum Analysis Report</h2>
      <div className={styles.tableWrapper}>
        <table className={styles.summaryTable}>
          <thead>
            <tr>
              <th>Frequency Range (MHz)</th>
              <th>Operator</th>
              <th>Signal Strength (dBm)</th>
              <th>Technology</th>
              <th>Service</th>
            </tr>
          </thead>
          <tbody>
            {data.map((item, index) => (
              <tr key={index} className={styles.tableRow}>
                <td>{item.frequency_range}</td>
                <td>{item.operator}</td>
                <td className={styles.strength}>
                  <span className={styles.value}>{item.strength} dB</span>
                  <div 
                    className={styles.strengthBar} 
                    style={{
                      width: `${Math.min(100, Math.max(0, (item.strength + 100) / 100 * 100))}%`
                    }}
                  />
                </td>
                <td>{item.technology}</td>
                <td>{item.service}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
} 