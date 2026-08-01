import axios from 'axios';
export const api = axios.create({ baseURL: import.meta.env.VITE_API_URL || 'http://127.0.0.1:5001', timeout: 60000 });
export async function predict(image) { const form = new FormData(); form.append('image', image); return (await api.post('/predict', form)).data; }