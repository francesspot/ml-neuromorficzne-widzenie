"use client";
import { useState } from "react";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setIsLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Błąd połączenia z API:", error);
      setResult({
        status: "error",
        message: "Nie udało się połączyć z backendem SNN.",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-50 flex items-center justify-center p-6">
      <div className="max-w-md w-full bg-white rounded-xl shadow-md p-8 border border-slate-100">
        <h1 className="text-2xl font-bold text-slate-800 mb-6 text-center">
          Neuromorficzne widzenie
        </h1>

        <form onSubmit={handleUpload} className="space-y-4">
          <div className="flex items-center justify-center w-full">
            <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-slate-300 border-dashed rounded-lg cursor-pointer bg-slate-50 hover:bg-slate-100 transition-colors">
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <p className="text-sm text-slate-500 font-medium">
                  {file ? file.name : "Wybierz plik z danymi"}
                </p>
              </div>
              <input
                type="file"
                className="hidden"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />
            </label>
          </div>

          <button
            type="submit"
            disabled={!file || isLoading}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 rounded-lg transition-colors disabled:opacity-50"
          >
            {isLoading ? "Analizowanie..." : "Rozpoznaj obiekt"}
          </button>
        </form>

        {result && (
          <div
            className={`mt-6 p-4 rounded-lg border ${result.status === "success" ? "bg-green-50 border-green-200 text-green-800" : "bg-red-50 border-red-200 text-red-800"}`}
          >
            <h3 className="font-semibold mb-1">Wynik inferencji:</h3>
            {result.status === "success" ? (
              <div>
                <p className="text-xl font-bold">{result.predicted_class}</p>
                <p className="text-sm opacity-75 mt-2">
                  Wymiary wgranego tensora:{" "}
                  {JSON.stringify(result.tensor_shape)}
                </p>
              </div>
            ) : (
              <p className="text-sm">{result.message}</p>
            )}
          </div>
        )}
      </div>
    </main>
  );
}
