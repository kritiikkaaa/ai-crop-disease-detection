import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import Uploader from "../components/Uploader";
import { predict } from "../lib/api";
import { savePrediction } from "../lib/history";
export default function Predict() {
  const [file, setFile] = useState(null),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const nav = useNavigate();
  const submit = async () => {
    if (!file) return setError("Choose a leaf image first.");
    setBusy(true);
    setError("");
    try {
      const result = await predict(file);
      const item = {
        ...result,
        image: URL.createObjectURL(file),
        at: new Date().toISOString(),
      };
      savePrediction(item);
      sessionStorage.setItem("latest-result", JSON.stringify(item));
      nav("/result");
    } catch (e) {
      setError(
        e.response?.data?.error ||
          "Could not analyze this image. Check the API connection and try again.",
      );
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="mx-auto max-w-2xl px-5 py-16">
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        <h1 className="text-4xl font-black">Analyze a crop leaf</h1>
        <p className="mt-3 text-slate-600 dark:text-slate-300">
          Use a focused, well-lit photo with one leaf filling most of the frame.
        </p>
        <div className="mt-8">
          <Uploader onSelect={setFile} />
        </div>
        {error && (
          <p className="mt-4 rounded-xl bg-red-50 p-4 text-sm text-red-700 dark:bg-red-950/40 dark:text-red-300">
            {error}
          </p>
        )}
        <button
          disabled={busy}
          onClick={submit}
          className="mt-6 w-full rounded-full bg-forest py-3.5 font-bold text-white disabled:opacity-60"
        >
          {busy ? "Analyzing with TensorFlow…" : "Get disease analysis"}
        </button>
      </motion.div>
    </section>
  );
}
