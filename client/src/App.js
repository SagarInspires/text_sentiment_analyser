// import { useState } from "react";
// import axios from "axios";

// function App() {
//   const [text, setText] = useState("");
//   const [result, setResult] = useState(null);

//   const analyze = async () => {
//     if (!text.trim()) return;

//     const res = await axios.post(
//       "http://localhost:5000/predict",
//       { text }
//     );

//     setResult(res.data);
//   };

//   return (
//     <div style={{ padding: "40px" }}>
//       <h2>Sentiment Analysis</h2>

//       <textarea
//         rows="5"
//         cols="60"
//         value={text}
//         onChange={(e) => setText(e.target.value)}
//         placeholder="Enter text"
//       />

//       <br /><br />

//       <button onClick={analyze}>Analyze</button>

//       {result && (
//         <div style={{ marginTop: "20px" }}>
//           <h3>
//             Sentiment:
//             {result.sentiment === 0 ? " Positive" : " Negative"}

//           </h3>
//           <p>Confidence: {result.confidence}</p>
//         </div>
//       )}
//     </div>
//   );
// }

// export default App;
import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyze = async () => {
    if (!text.trim()) {
      setError("Please enter some text to analyze");
      return;
    }

    try {
      setError("");
      setLoading(true);
      const res = await axios.post("http://localhost:5000/predict", { text });
      setResult(res.data);
    } catch (err) {
      setError("Unable to reach server. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="card">
        <h2 className="title">Sentiment Analysis</h2>

        <textarea
          className="input"
          rows="5"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type or paste text here..."
        />

        <button className="btn" onClick={analyze} disabled={loading}>
          {loading ? "Analyzing…" : "Analyze"}
        </button>

        {error && <p className="error">{error}</p>}

        {result && (
          <div className={`result ${result.sentiment === 0 ? "positive" : "negative"}`}>
            <h3>
              {result.sentiment === 0 ? "Positive" : "Negative"}
            </h3>
            <p>Confidence: {100*(result.confidence)+'%'}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
