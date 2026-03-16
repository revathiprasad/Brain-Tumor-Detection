import React, { useState } from "react";
import axios from "axios";
import "./index.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [model, setModel] = useState("cnn");
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState("");
  const [modelUsed, setModelUsed] = useState("");
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setPrediction("");
    setConfidence("");
    setModelUsed("");
  };

  const handlePredict = async () => {
    if (!file) {
      alert("Please upload an image");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("model", model);

    setLoading(true);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/predict",
        formData
      );

      setPrediction(response.data.prediction);
      setConfidence(response.data.confidence + "%");
      setModelUsed(response.data.model_used);
    } catch (error) {
      alert("Prediction failed");
    }

    setLoading(false);
  };

  return (
    <div className="app">
      <div className="card">
        <h1>🧠 Brain Tumor Detection</h1>
        <p className="subtitle">
          Upload an MRI image and select a model to predict
        </p>

        <label className="upload-box">
          <input type="file" accept="image/*" onChange={handleFileChange} />
          <span>📤 Click to upload MRI image</span>
        </label>

        {preview && <img src={preview} alt="preview" className="preview" />}

        <div className="model-select">
          <label className={model === "cnn" ? "active" : ""}>
            <input
              type="radio"
              value="cnn"
              checked={model === "cnn"}
              onChange={() => setModel("cnn")}
            />
            CNN
          </label>

          <label className={model === "cnn_aug" ? "active" : ""}>
            <input
              type="radio"
              value="cnn_aug"
              checked={model === "cnn_aug"}
              onChange={() => setModel("cnn_aug")}
            />
            CNN + Aug
          </label>

          <label className={model === "vgg" ? "active" : ""}>
            <input
              type="radio"
              value="vgg"
              checked={model === "vgg"}
              onChange={() => setModel("vgg")}
            />
            VGG16
          </label>
        </div>

        <button className="predict-btn" onClick={handlePredict} disabled={loading}>
          {loading ? "🔍 Predicting..." : "🚀 Predict"}
        </button>

        {prediction && (
          <div className="result">
            <h3>Prediction: <span>{prediction}</span></h3>
            <h4>Confidence: <span>{confidence}</span></h4>
            <p>Model Used: <b>{modelUsed}</b></p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
