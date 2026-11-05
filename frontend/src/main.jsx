import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const initialForm = {
  total_sqft: "1200",
  bhk: "2",
  bath: "2",
  location: "",
};

function App() {
  const [form, setForm] = useState(initialForm);
  const [locations, setLocations] = useState([]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch("/api/model-info")
      .then((response) => {
        if (!response.ok) throw new Error("Could not load locations.");
        return response.json();
      })
      .then((data) => {
        const availableLocations = data.locations ?? [];
        setLocations(availableLocations);
        setForm((current) => ({
          ...current,
          location: availableLocations[0] ?? "",
        }));
      })
      .catch(() => {
        setError(
          "Start the backend at http://127.0.0.1:8000 to load locations.",
        );
      });
  }, []);

  function handleChange(event) {
    setForm({ ...form, [event.target.name]: event.target.value });
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setResult(null);
    setLoading(true);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          total_sqft: Number(form.total_sqft),
          bhk: Number(form.bhk),
          bath: Number(form.bath),
          location: form.location,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Prediction failed.");
      setResult(data);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="app-card">
        <div className="intro">
          <p className="eyebrow">Bangalore real estate</p>
          <h1>Find your home&apos;s estimated value.</h1>
          <p className="subtitle">
            Enter a few details to get an instant machine-learning estimate.
          </p>
        </div>

        <div className="content-grid">
          <section className="panel input-panel">
            <div className="panel-heading">
              <p className="panel-kicker">Step 1</p>
              <h2>Your property details</h2>
              <p>Enter the details below to calculate an estimated value.</p>
            </div>

            <form className="prediction-form" onSubmit={handleSubmit}>
              <label>
                Location
                <select
                  name="location"
                  value={form.location}
                  onChange={handleChange}
                  required
                  disabled={!locations.length}>
                  <option value="" disabled>
                    {locations.length
                      ? "Select a location"
                      : "Loading locations..."}
                  </option>
                  {locations.map((location) => (
                    <option key={location} value={location}>
                      {location}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                Area <span>sq. ft.</span>
                <input
                  name="total_sqft"
                  type="number"
                  min="100"
                  step="1"
                  value={form.total_sqft}
                  onChange={handleChange}
                  required
                />
              </label>

              <div className="field-row">
                <label>
                  Bedrooms
                  <input
                    name="bhk"
                    type="number"
                    min="1"
                    max="20"
                    step="1"
                    value={form.bhk}
                    onChange={handleChange}
                    required
                  />
                </label>
                <label>
                  Bathrooms
                  <input
                    name="bath"
                    type="number"
                    min="1"
                    max="20"
                    step="1"
                    value={form.bath}
                    onChange={handleChange}
                    required
                  />
                </label>
              </div>

              <button type="submit" disabled={loading || !locations.length}>
                {loading ? "Estimating..." : "Estimate price"}
              </button>
            </form>
          </section>

          <section className="panel result-panel" aria-live="polite">
            <div className="panel-heading">
              <p className="panel-kicker">Step 2</p>
              <h2>Prediction result</h2>
              <p>Your estimated house price will appear here.</p>
            </div>

            {error && <p className="message error">{error}</p>}

            {result ? (
              <div className="result">
                <div>
                  <p className="result-label">Estimated price</p>
                  <p className="price">
                    ₹
                    {result.predicted_price.toLocaleString("en-IN", {
                      maximumFractionDigits: 2,
                    })}{" "}
                    <span>lakhs</span>
                  </p>
                </div>
              </div>
            ) : (
              <div className="result-empty">
                <span className="result-icon" aria-hidden="true">
                  ₹
                </span>
                <p>
                  Complete the form and select “Estimate price” to see your
                  result.
                </p>
              </div>
            )}
          </section>
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
