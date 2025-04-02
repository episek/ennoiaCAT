import { ApiResponse } from '../types/spectrum';

export const fetchSpectrumData = async (): Promise<ApiResponse> => {
  try {
    const response = await fetch('http://localhost:8000/prompt', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_message: "analyze spectrum"
      })
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    return await response.json();
  } catch (error) {
    console.error('Failed to fetch spectrum data:', error);
    throw error;
  }
}; 