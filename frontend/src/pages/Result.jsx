import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

const displayName = (value) => value.replace("___", " — ").replaceAll("_", " ");
const percent = (value) => `${(Number(value) * 100).toFixed(1)}%`;

export default function Result() {
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const nav = useNavigate();

  useEffect(() => {
    const storedResult = sessionStorage.getItem("latest-result");
    if (!storedResult) {
      setError("No prediction available. Upload a leaf image first.");
      return;
    }
    try {
      setResult(JSON.parse(storedResult));
    } catch (e) {
      setError("Failed to load prediction result.");
    }
  }, []);

  if (error) {
    return (
      <section className="mx-auto max-w-3xl px-5 py-20">
        <p className="rounded-2xl bg-red-50 p-5 text-red-700 dark:bg-red-950/40 dark:text-red-300">
          {error}
        </p>
        <button
          onClick={() => nav("/predict")}
          className="mt-6 rounded-full bg-forest px-8 py-3 font-bold text-white"
        >
          Make a new prediction
        </button>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="px-5 py-20 text-center">Loading prediction results…</section>
    );
  }

  const { status, image, confidence, prediction, crop, disease, symptoms, causes, treatment, prevention, top_predictions, message, normalized_entropy } = result;

  const isUnknown = status === "unknown";

  return (
    <motion.section
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="mx-auto max-w-3xl px-5 py-12"
    >
      {/* Image & Result Header */}
      <div className="glass rounded-2xl p-8">
        <div className="grid gap-8 lg:grid-cols-2">
          {/* Image */}
          <div className="flex items-center justify-center">
            <img
              src={image}
              alt="Uploaded leaf"
              className="w-full rounded-xl object-cover"
            />
          </div>

          {/* Result Details */}
          <div className="flex flex-col justify-center">
            {isUnknown ? (
              <>
                <p className="text-sm font-medium uppercase tracking-widest text-orange-600 dark:text-orange-400">
                  Unknown / Low Confidence
                </p>
                <h1 className="mt-2 text-3xl font-black text-slate-900 dark:text-white">
                  Cannot classify
                </h1>
                <p className="mt-4 text-slate-600 dark:text-slate-300">
                  {message}
                </p>
                <div className="mt-6 space-y-2 rounded-xl bg-orange-50 p-4 dark:bg-orange-950/30">
                  <p className="text-sm font-medium text-orange-900 dark:text-orange-200">
                    Model Confidence: {confidence}%
                  </p>
                  <p className="text-xs text-orange-700 dark:text-orange-300">
                    Entropy: {normalized_entropy}
                  </p>
                </div>
              </>
            ) : (
              <>
                <p className="text-sm font-medium uppercase tracking-widest text-forest dark:text-lime">
                  Prediction
                </p>
                <h1 className="mt-2 text-3xl font-black text-slate-900 dark:text-white">
                  {displayName(prediction)}
                </h1>
                <p className="mt-2 text-slate-600 dark:text-slate-300">
                  {crop && <span className="font-medium">{displayName(crop)}</span>}
                </p>
                <p className="mt-4 text-slate-600 dark:text-slate-300">
                  {message}
                </p>
                <div className="mt-6 rounded-xl bg-emerald-50 p-4 dark:bg-emerald-950/30">
                  <p className="text-sm text-slate-600 dark:text-slate-300">
                    Model Confidence
                  </p>
                  <p className="mt-1 text-3xl font-black text-forest dark:text-lime">
                    {confidence}%
                  </p>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Disease Information */}
      {!isUnknown && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mt-8"
        >
          <h2 className="text-2xl font-bold">Disease information</h2>
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            {symptoms && symptoms.length > 0 && (
              <div className="glass rounded-xl p-4">
                <h3 className="font-bold text-forest dark:text-lime">Symptoms</h3>
                <ul className="mt-2 space-y-1 text-sm text-slate-700 dark:text-slate-300">
                  {symptoms.map((item, i) => (
                    <li key={i} className="flex gap-2">
                      <span>•</span>
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {causes && causes.length > 0 && (
              <div className="glass rounded-xl p-4">
                <h3 className="font-bold text-forest dark:text-lime">Causes</h3>
                <ul className="mt-2 space-y-1 text-sm text-slate-700 dark:text-slate-300">
                  {causes.map((item, i) => (
                    <li key={i} className="flex gap-2">
                      <span>•</span>
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {treatment && treatment.length > 0 && (
              <div className="glass rounded-xl p-4">
                <h3 className="font-bold text-forest dark:text-lime">Treatment</h3>
                <ul className="mt-2 space-y-1 text-sm text-slate-700 dark:text-slate-300">
                  {treatment.map((item, i) => (
                    <li key={i} className="flex gap-2">
                      <span>•</span>
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {prevention && prevention.length > 0 && (
              <div className="glass rounded-xl p-4">
                <h3 className="font-bold text-forest dark:text-lime">Prevention</h3>
                <ul className="mt-2 space-y-1 text-sm text-slate-700 dark:text-slate-300">
                  {prevention.map((item, i) => (
                    <li key={i} className="flex gap-2">
                      <span>•</span>
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </motion.div>
      )}

      {/* Top Predictions */}
      {!isUnknown && top_predictions && top_predictions.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mt-8"
        >
          <h2 className="text-2xl font-bold">Top predictions</h2>
          <div className="mt-4 space-y-2">
            {top_predictions.map((pred, i) => (
              <div
                key={i}
                className="glass flex items-center justify-between rounded-xl p-4"
              >
                <span className="font-medium text-slate-900 dark:text-white">
                  {displayName(pred.disease)}
                </span>
                <span className="text-sm font-bold text-slate-600 dark:text-slate-400">
                  {pred.confidence}%
                </span>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Disclaimer */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="mt-8 rounded-xl bg-blue-50 p-4 text-sm text-blue-700 dark:bg-blue-950/30 dark:text-blue-300"
      >
        <p className="font-medium">⚠️ Disclaimer</p>
        <p className="mt-1">
          This AI-assisted tool is for screening purposes only and is not a substitute for professional agricultural diagnosis. Always consult local agricultural extension services or experts before making treatment decisions.
        </p>
      </motion.div>

      {/* Action Buttons */}
      <div className="mt-8 flex gap-4">
        <button
          onClick={() => nav("/predict")}
          className="flex-1 rounded-full bg-forest py-3 font-bold text-white"
        >
          Analyze another leaf
        </button>
        <button
          onClick={() => nav("/results")}
          className="flex-1 rounded-full border-2 border-forest bg-transparent py-3 font-bold text-forest dark:border-lime dark:text-lime"
        >
          View results dashboard
        </button>
      </div>
    </motion.section>
  );
}
