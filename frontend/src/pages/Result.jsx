import { Link } from "react-router-dom";
import { FiCopy, FiDownload, FiShare2 } from "react-icons/fi";
import { jsPDF } from "jspdf";
const List = ({ title, items }) => (
  <section>
    <h2 className="font-bold text-forest dark:text-lime">{title}</h2>
    <ul className="mt-2 list-disc space-y-1 pl-5 text-slate-600 dark:text-slate-300">
      {items?.map((x) => (
        <li key={x}>{x}</li>
      ))}
    </ul>
  </section>
);
export default function Result() {
  const raw = sessionStorage.getItem("latest-result");
  if (!raw)
    return (
      <section className="px-5 py-24 text-center">
        <p>No recent result found.</p>
        <Link className="mt-4 inline-block text-moss" to="/predict">
          Upload an image
        </Link>
      </section>
    );
  const r = JSON.parse(raw);
  const copy = () =>
    navigator.clipboard.writeText(`${r.disease} — ${r.confidence}% confidence`);
  const pdf = () => {
    const d = new jsPDF();
    d.setFontSize(20);
    d.text("CropVision Report", 20, 20);
    d.setFontSize(13);
    d.text(`Result: ${r.disease}`, 20, 35);
    d.text(`Confidence: ${r.confidence}%`, 20, 44);
    d.text(`Treatment: ${(r.treatment || []).join("; ")}`, 20, 55, {
      maxWidth: 170,
    });
    d.text(`Prevention: ${(r.prevention || []).join("; ")}`, 20, 75, {
      maxWidth: 170,
    });
    d.save("cropvision-report.pdf");
  };
  return (
    <section className="mx-auto max-w-4xl px-5 py-12">
      <div id="result" className="glass rounded-3xl p-7 md:p-10">
        <div className="grid gap-7 md:grid-cols-[1fr_auto]">
          <div>
            <p className="font-bold uppercase tracking-widest text-moss">
              Model result
            </p>
            <h1 className="mt-2 text-4xl font-black">{r.disease}</h1>
            <p className="mt-4 text-slate-600 dark:text-slate-300">
              {r.description}
            </p>
            <p className="mt-4 text-sm italic text-slate-500">
              Scientific name: {r.scientific_name}
            </p>
          </div>
          <div className="flex h-36 w-36 items-center justify-center rounded-full border-[12px] border-lime bg-white text-center dark:bg-slate-950">
            <strong className="text-3xl text-forest dark:text-lime">
              {r.confidence}%
            </strong>
          </div>
        </div>
        <div className="mt-9 grid gap-7 md:grid-cols-2">
          <List title="Symptoms" items={r.symptoms} />
          <List title="Likely causes" items={r.causes} />
          <List title="Treatment guidance" items={r.treatment} />
          <List title="Prevention" items={r.prevention} />
        </div>
      </div>
      <div className="mt-6 flex flex-wrap gap-3">
        <Link
          to="/predict"
          className="rounded-full bg-forest px-5 py-2.5 font-bold text-white"
        >
          Upload again
        </Link>
        <button onClick={copy} className="action">
          <FiCopy /> Copy
        </button>
        <button onClick={pdf} className="action">
          <FiDownload /> PDF report
        </button>
        <button
          onClick={() =>
            navigator.share?.({
              title: "CropVision result",
              text: `${r.disease}: ${r.confidence}%`,
            })
          }
          className="action"
        >
          <FiShare2 /> Share
        </button>
      </div>
    </section>
  );
}
