import { useEffect, useState } from "react";
import { getResults, graphUrl } from "../lib/api";

const percent = (value) => `${(Number(value) * 100).toFixed(2)}%`;
const displayName = (value) => value.replace("___", " — ").replaceAll("_", " ");

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getResults().then(setData).catch((requestError) => {
      setError(requestError.response?.data?.error || "Unable to load generated model results.");
    });
  }, []);

  if (error) return <section className="mx-auto max-w-3xl px-5 py-20"><p className="rounded-2xl bg-red-50 p-5 text-red-700">{error}</p></section>;
  if (!data) return <section className="px-5 py-20 text-center">Loading real evaluation results…</section>;

  const metricCards = [
    ["Test accuracy", data.metrics.accuracy],
    ["Macro precision", data.metrics.macro_precision],
    ["Macro recall", data.metrics.macro_recall],
    ["Macro F1", data.metrics.macro_f1],
    ["Weighted precision", data.metrics.weighted_precision],
    ["Weighted recall", data.metrics.weighted_recall],
    ["Weighted F1", data.metrics.weighted_f1],
  ];

  return (
    <section className="mx-auto max-w-7xl px-5 py-12">
      <p className="font-bold uppercase tracking-widest text-moss">Generated evaluation</p>
      <h1 className="mt-2 text-4xl font-black">Model results dashboard</h1>
      <p className="mt-3 text-slate-600 dark:text-slate-300">
        Metrics and figures below are loaded from the trained model’s saved test evaluation.
      </p>
      <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {metricCards.map(([label, value]) => (
          <article className="glass rounded-2xl p-5" key={label}>
            <p className="text-sm text-slate-500">{label}</p>
            <p className="mt-2 text-3xl font-black text-forest dark:text-lime">{percent(value)}</p>
          </article>
        ))}
      </div>
      <p className="mt-5 text-sm text-slate-500">Held-out test images: {data.metrics.test_image_count}. Classes: {data.model_metadata?.class_count ?? "—"}.</p>

      <h2 className="mt-14 text-2xl font-bold">Generated graphs</h2>
      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        {data.graphs.map((filename) => (
          <figure className="glass overflow-hidden rounded-2xl p-3" key={filename}>
            <img src={graphUrl(filename)} alt={filename.replaceAll("_", " ")} className="w-full rounded-xl" />
            <figcaption className="p-2 text-sm font-medium capitalize">{filename.replace(".png", "").replaceAll("_", " ")}</figcaption>
          </figure>
        ))}
      </div>

      <h2 className="mt-14 text-2xl font-bold">Class-wise test metrics</h2>
      <div className="mt-5 overflow-x-auto rounded-2xl border border-emerald-900/10 bg-white dark:bg-slate-900">
        <table className="w-full min-w-[720px] text-left text-sm">
          <thead className="bg-emerald-50 text-forest dark:bg-emerald-950/40 dark:text-lime">
            <tr><th className="p-4">Class</th><th className="p-4">Precision</th><th className="p-4">Recall</th><th className="p-4">F1-score</th><th className="p-4">Support</th></tr>
          </thead>
          <tbody>
            {data.class_metrics.map((item) => (
              <tr className="border-t border-emerald-900/10" key={item.class}>
                <td className="p-4 font-medium">{displayName(item.class)}</td>
                <td className="p-4">{percent(item.precision)}</td>
                <td className="p-4">{percent(item.recall)}</td>
                <td className="p-4">{percent(item["f1-score"])}</td>
                <td className="p-4">{item.support}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
