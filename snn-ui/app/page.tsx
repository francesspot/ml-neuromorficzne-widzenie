"use client";

import { useState } from "react";

const NCALTECH_CLASSES = [
  "BACKGROUND_Google",
  "Faces_easy",
  "Leopards",
  "Motorbikes",
  "accordion",
  "airplanes",
  "anchor",
  "ant",
  "barrel",
  "bass",
  "beaver",
  "binocular",
  "bonsai",
  "brain",
  "brontosaurus",
  "buddha",
  "butterfly",
  "camera",
  "cannon",
  "car_side",
  "ceiling_fan",
  "cellphone",
  "chair",
  "chandelier",
  "cougar_body",
  "cougar_face",
  "crab",
  "crayfish",
  "crocodile",
  "crocodile_head",
  "cup",
  "dalmatian",
  "dollar_bill",
  "dolphin",
  "dragonfly",
  "electric_guitar",
  "elephant",
  "emu",
  "euphonium",
  "ewer",
  "ferry",
  "flamingo",
  "flamingo_head",
  "garfield",
  "gerenuk",
  "gramophone",
  "grand_piano",
  "hawksbill",
  "headphone",
  "hedgehog",
  "helicopter",
  "ibis",
  "inline_skate",
  "joshua_tree",
  "kangaroo",
  "ketch",
  "lamp",
  "laptop",
  "llama",
  "lobster",
  "lotus",
  "mandolin",
  "mayfly",
  "menorah",
  "metronome",
  "minaret",
  "nautilus",
  "octopus",
  "okapi",
  "pagoda",
  "panda",
  "pigeon",
  "pizza",
  "platypus",
  "pyramid",
  "revolver",
  "rhino",
  "rooster",
  "saxophone",
  "schooner",
  "scissors",
  "scorpion",
  "sea_horse",
  "snoopy",
  "soccer_ball",
  "stapler",
  "starfish",
  "stegosaurus",
  "stop_sign",
  "strawberry",
  "sunflower",
  "tick",
  "trilobite",
  "umbrella",
  "watch",
  "water_lilly",
  "wheelchair",
  "wild_cat",
  "windsor_chair",
  "wrench",
  "yin_yang",
];

const MODEL_STATS: Record<string, any> = {
  franciszek_scnn: {
    name: "FranciszekSCNN (Rate Coding)",
    acc: "53.1%",
    latency: "~120ms",
    mem: "150 MB",
    train: "3.5h",
  },
  weronika_temp: {
    name: "Weronika SNN (Temporal Coding)",
    acc: "TBD",
    latency: "TBD",
    mem: "TBD",
    train: "TBD",
  },
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [selectedModel, setSelectedModel] = useState("franciszek_scnn");
  const [showCorrection, setShowCorrection] = useState(false);
  const [correctClass, setCorrectClass] = useState(NCALTECH_CLASSES[0]);
  const [isRetraining, setIsRetraining] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
      setResult(null);
      setShowCorrection(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);
    setShowCorrection(false);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Błąd serwera. Upewnij się, że FastAPI działa.");
      }

      const data = await response.json();
      if (data.status === "success") {
        setResult(data);
      } else {
        setError(data.message);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRetrain = async () => {
    if (!file) return;

    setIsRetraining(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("correct_class", correctClass);

    try {
      const res = await fetch("http://localhost:8000/retrain", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (data.status === "success") {
        alert(`Sukces! ${data.message}`);
        setShowCorrection(false);
      } else {
        alert(`Błąd: ${data.message}`);
      }
    } catch (err) {
      console.error(err);
      alert("Błąd połączenia z serwerem.");
    } finally {
      setIsRetraining(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-12 bg-gray-50 text-gray-900">
      <h1 className="text-4xl font-bold mb-8">Neuromorficzne Widzenie - SNN</h1>

      <div className="bg-white p-6 rounded-xl shadow-md w-full max-w-md mb-6">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Wybierz model SNN:
        </label>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="block w-full p-2 border border-gray-300 rounded mb-4 focus:ring-blue-500 focus:border-blue-500"
        >
          <option value="franciszek_scnn">
            {MODEL_STATS.franciszek_scnn.name}
          </option>
          <option value="weronika_temp" disabled>
            {MODEL_STATS.weronika_temp.name} - (Wkrótce)
          </option>
        </select>

        <div className="grid grid-cols-2 gap-y-2 gap-x-4 text-sm text-gray-600 bg-gray-50 p-3 rounded border">
          <div>
            <strong className="text-gray-800">Accuracy:</strong>{" "}
            {MODEL_STATS[selectedModel].acc}
          </div>
          <div>
            <strong className="text-gray-800">Predykcja:</strong>{" "}
            {MODEL_STATS[selectedModel].latency}
          </div>
          <div>
            <strong className="text-gray-800">Pamięć:</strong>{" "}
            {MODEL_STATS[selectedModel].mem}
          </div>
          <div>
            <strong className="text-gray-800">Trening:</strong>{" "}
            {MODEL_STATS[selectedModel].train}
          </div>
        </div>
      </div>

      <div className="bg-white p-8 rounded-xl shadow-md w-full max-w-md">
        <form
          onSubmit={handleSubmit}
          className="flex flex-col gap-6 items-center"
        >
          <label className="w-full">
            <span className="block text-sm font-medium text-gray-700 mb-2">
              Wybierz nagranie wideo (.mp4)
            </span>
            <input
              type="file"
              accept=".mp4"
              onChange={handleFileChange}
              className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
          </label>
          <button
            type="submit"
            disabled={!file || loading}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded transition-colors disabled:bg-gray-400"
          >
            {loading ? "Przetwarzanie przez SNN..." : "Rozpoznaj obiekt"}
          </button>
        </form>
      </div>

      {error && <p className="text-red-500 mt-6 font-semibold">{error}</p>}

      {result && (
        <div className="mt-12 flex flex-col items-center gap-6 w-full max-w-4xl bg-white p-8 rounded-xl shadow-md">
          <div className="text-center">
            <h2 className="text-3xl font-semibold text-green-600">
              Wykryta klasa: {result.predicted_class}
            </h2>
          </div>

          <div className="w-full max-w-md">
            {!showCorrection ? (
              <div className="text-center">
                <button
                  onClick={() => setShowCorrection(true)}
                  className="text-sm text-red-500 underline hover:text-red-700 transition-colors"
                >
                  Zła klasa? Popraw wynik i dotrenuj model
                </button>
              </div>
            ) : (
              <div className="mt-2 p-5 border-2 border-red-100 bg-red-50 rounded-xl flex flex-col gap-4 shadow-inner">
                <label className="text-sm font-semibold text-gray-800">
                  Wskaż prawidłową klasę dla tego nagrania:
                </label>
                <select
                  value={correctClass}
                  onChange={(e) => setCorrectClass(e.target.value)}
                  className="p-2 border border-gray-300 rounded focus:ring-red-500 focus:border-red-500 bg-white"
                >
                  {NCALTECH_CLASSES.map((cls) => (
                    <option key={cls} value={cls}>
                      {cls}
                    </option>
                  ))}
                </select>
                <button
                  onClick={handleRetrain}
                  disabled={isRetraining}
                  className="bg-red-600 text-white font-bold py-2 rounded hover:bg-red-700 transition-colors disabled:bg-gray-400"
                >
                  {isRetraining
                    ? "Trwa mikro-trening..."
                    : "Zatwierdź i dotrenuj model"}
                </button>
                <button
                  onClick={() => setShowCorrection(false)}
                  className="text-xs text-gray-500 underline hover:text-gray-700 text-center"
                >
                  Anuluj
                </button>
              </div>
            )}
          </div>

          {result.animation_base64 && (
            <div className="mt-6 flex flex-col items-center w-full">
              <h3 className="text-lg font-medium text-gray-900 mb-4">
                Podgląd: Oryginał vs. Emulator kamery DVS
              </h3>
              <img
                src={result.animation_base64}
                alt="Symulacja wizji neuromorficznej"
                className="border-2 border-gray-200 rounded-lg shadow-sm w-full object-contain"
              />
            </div>
          )}
        </div>
      )}
    </main>
  );
}
