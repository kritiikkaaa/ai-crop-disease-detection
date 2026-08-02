import { useRef, useState } from "react";
import { FiUploadCloud, FiX } from "react-icons/fi";
export default function Uploader({ onSelect }) {
  const [preview, setPreview] = useState(null);
  const input = useRef();
  const choose = (file) => {
    if (!file?.type.startsWith("image/")) return;
    setPreview(URL.createObjectURL(file));
    onSelect(file);
  };
  const clear = () => {
    setPreview(null);
    onSelect(null);
    if (input.current) input.current.value = "";
  };
  return (
    <div
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        choose(e.dataTransfer.files[0]);
      }}
      className="rounded-3xl border-2 border-dashed border-moss/40 bg-white/60 p-7 text-center dark:bg-slate-900/60"
    >
      {preview ? (
        <div className="relative mx-auto max-w-md">
          <img
            src={preview}
            className="max-h-72 w-full rounded-2xl object-cover"
          />
          <button
            onClick={clear}
            className="absolute right-3 top-3 rounded-full bg-slate-950/75 p-2 text-white"
          >
            <FiX />
          </button>
        </div>
      ) : (
        <>
          <FiUploadCloud className="mx-auto mb-3 text-5xl text-moss" />
          <p className="font-semibold">Drop a clear leaf photo here</p>
          <p className="mt-1 text-sm text-slate-500">
            PNG, JPG, or WEBP · 8 MB maximum
          </p>
          <button
            onClick={() => input.current.click()}
            className="mt-5 rounded-full bg-forest px-5 py-2.5 font-semibold text-white"
          >
            Browse image
          </button>
          <input
            ref={input}
            className="hidden"
            type="file"
            accept="image/png,image/jpeg,image/webp"
            onChange={(e) => choose(e.target.files[0])}
          />
        </>
      )}
    </div>
  );
}
