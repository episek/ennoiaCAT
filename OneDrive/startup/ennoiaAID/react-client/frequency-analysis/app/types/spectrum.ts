export interface SpectrumData {
  operator: string;
  strength: number;
  technology: string;
  service: string;
  frequency_range: string;
}


export interface ApiResponse {
  signal_strengths: number[];
  frequency: number[];
  frequency_report: {
    frequency_ranges: SpectrumData[];
  };
} 