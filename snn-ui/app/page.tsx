"use client";

import { useState } from "react";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);

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

  return (
    <main className="flex min-h-screen flex-col items-center p-12 bg-gray-50 text-gray-900">
      <h1 className="text-4xl font-bold mb-8">Neuromorficzne Widzenie - SNN</h1>

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
          <h2 className="text-3xl font-semibold text-green-600">
            Wykryta klasa: {result.predicted_class}
          </h2>

          {result.animation_base64 && (
            <div className="mt-4 flex flex-col items-center w-full">
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
