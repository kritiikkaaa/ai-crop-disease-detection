const key = "crop-vision-history";
export const loadHistory = () => JSON.parse(localStorage.getItem(key) || "[]");
export const savePrediction = (item) =>
  localStorage.setItem(
    key,
    JSON.stringify([item, ...loadHistory()].slice(0, 8)),
  );
