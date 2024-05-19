import React, { useState, useRef, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import "./App.css";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import ModelVisualizer from "./ModelVisualizer";
import TrainingCharts from "./TrainingCharts";
import {
  createAlexNet,
  createModel,
  createDecoyModel,
  createMobileNetModel,
  createSimpleModel,
  createComplexModel,
} from "./Models/Alexnet";
import JSZip, { forEach } from "jszip";
import { saveAs } from "file-saver";
import "canvas-toBlob";

function App() {
  const [isDrawing, setIsDrawing] = useState(false);
  const [dataset, setDataset] = useState([]);
  const [model, setModel] = useState(null);
  const [labels, setLabels] = useState([]);
  const drawCanvasRef = useRef(null);
  const inferenceCanvasRef = useRef(null);
  const predictionOutputRef = useRef(null);
  const labelSelectRef = useRef(null);
  const modelCreators = {
    SimpleCNN: createSimpleModel,
    ComplexCNN: createComplexModel,
    AlexNet: createAlexNet,
    DecoyModel: createDecoyModel,
    // Add more models as needed
  };
  const [modelList, setModelList] = useState(Object.keys(modelCreators));
  const [selectedModel, setSelectedModel] = useState(modelList[0]);
  const [isTraining, setIsTraining] = useState(false);
  const [trainLog, setTrainLog] = useState("");
  const [showLog, setShowLog] = useState(false);
  const [epochs, setEpochs] = useState(20);
  const [animationPlay, setAnimationPlay] = useState(false);
  const [trainingLogs, setTrainingLogs] = useState([]);
  const fileInputRef = useRef(null);
  const inputElementRef = useRef(null);
  const [predictionOutput, setPredictionOutput] = useState([]);
  const predictionProbabilitiesRef = useRef(null);

  const handleButtonClick = () => {
    fileInputRef.current.click();
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    try {
      loadDatasetFromZip(file);
      inputElementRef.current.value = "";
    } catch (error) {
      alert("Error occured when loading the dataset");
    }
  };

  useEffect(() => {
    const drawCanvas = drawCanvasRef.current;
    const inferenceCanvas = inferenceCanvasRef.current;
    const drawContext = drawCanvas.getContext("2d");
    const inferenceContext = inferenceCanvas.getContext("2d");

    drawContext.lineWidth = 10;
    inferenceContext.lineWidth = 10;
    drawContext.fillStyle = "black";
    drawContext.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    inferenceContext.fillStyle = "black";
    inferenceContext.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  }, []);

  const startDrawing = (e) => {
    setIsDrawing(true);
    draw(e);
  };

  const draw = (e) => {
    if (!isDrawing) return;
    const drawCanvas = drawCanvasRef.current;
    const drawContext = drawCanvas.getContext("2d");
    const rect = drawCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Fill with gradient
    drawContext.strokeStyle = "white";

    drawContext.lineTo(x, y);
    drawContext.stroke();
    drawContext.beginPath();
    drawContext.moveTo(x, y);
  };

  const stopDrawing = () => {
    setIsDrawing(false);
    const drawCanvas = drawCanvasRef.current;
    const drawContext = drawCanvas.getContext("2d");
    drawContext.beginPath();
  };

  const startInferenceDrawing = (e) => {
    setIsDrawing(true);
    inferenceDraw(e);
  };

  const inferenceDraw = (e) => {
    if (!isDrawing) return;
    const inferenceCanvas = inferenceCanvasRef.current;
    const inferenceContext = inferenceCanvas.getContext("2d");
    const rect = inferenceCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    inferenceContext.strokeStyle = "white";

    inferenceContext.lineTo(x, y);
    inferenceContext.stroke();
    inferenceContext.beginPath();
    inferenceContext.moveTo(x, y);
  };

  const stopInferenceDrawing = async () => {
    setIsDrawing(false);
    const inferenceCanvas = inferenceCanvasRef.current;
    const inferenceContext = inferenceCanvas.getContext("2d");
    inferenceContext.beginPath();
    await performPrediction();
  };

  const performPrediction = async () => {
    const inferenceCanvas = inferenceCanvasRef.current;
    const inferenceContext = inferenceCanvas.getContext("2d");
    const imageData = inferenceContext.getImageData(0, 0, 280, 280);
    const tensor = tf.browser
      .fromPixels(imageData, 1)
      .resizeNearestNeighbor([102, 102])
      .toFloat()
      .div(tf.scalar(255))
      .reshape([1, 102, 102, 1]);

    const prediction = await model.predict(tensor).data();
    console.log(prediction);
    const predictedIndex = prediction.indexOf(Math.max(...prediction));
    const predictedLabel = labels[predictedIndex];
    setPredictionOutput(prediction);
    predictionOutputRef.current.innerText = predictedLabel;

    // Clear previous probabilities
    predictionProbabilitiesRef.current.innerHTML = "";

    // Display all prediction probabilities
    prediction.forEach((probability, index) => {
      const probElement = document.createElement("p");
      probElement.innerText = `${labels[index]}: ${(probability * 100).toFixed(
        2
      )}%`;
      predictionProbabilitiesRef.current.appendChild(probElement);
    });

    console.log(predictionOutput);
  };

  const clearDrawCanvas = () => {
    const drawCanvas = drawCanvasRef.current;
    const drawContext = drawCanvas.getContext("2d");
    drawContext.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    drawContext.fillStyle = "black";
    drawContext.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  };

  const addDataset = () => {
    const drawCanvas = drawCanvasRef.current;
    const drawContext = drawCanvas.getContext("2d");
    const labelIndex = labelSelectRef.current.selectedIndex;
    const imageData = drawContext.getImageData(
      0,
      0,
      drawCanvas.width,
      drawCanvas.height
    );
    const tensor = tf.browser
      .fromPixels(imageData, 1)
      .resizeNearestNeighbor([102, 102])
      .toFloat()
      .div(tf.scalar(255));
    setDataset([...dataset, { xs: tensor, ys: labelIndex }]);
    drawContext.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    drawContext.fillStyle = "black";
    drawContext.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    toast("Data added!");
  };

  const tensorToImageData = async (tensor) => {
    const [height, width] = [tensor.shape[0], tensor.shape[1]];
    const data = await tensor.data();
    const clampedArray = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < data.length; i++) {
      const value = data[i] * 255;
      clampedArray[i * 4] = value;
      clampedArray[i * 4 + 1] = value;
      clampedArray[i * 4 + 2] = value;
      clampedArray[i * 4 + 3] = 255;
    }
    return new ImageData(clampedArray, width, height);
  };

  const saveDatasetToZip = async () => {
    console.log("Function executed");
    const zip = new JSZip();

    const createZipEntry = (data, label, index) => {
      return new Promise(async (resolve) => {
        // Create a canvas to draw the image data
        const canvas = document.createElement("canvas");
        canvas.width = 102;
        canvas.height = 102;
        const context = canvas.getContext("2d");

        // Convert the tensor to an image
        const imageData = await tensorToImageData(data.xs);
        context.putImageData(imageData, 0, 0);

        // Convert canvas to Blob
        canvas.toBlob((blob) => {
          zip.folder(label).file(`${index}.png`, blob);
          resolve();
        });
      });
    };

    // Process all dataset entries
    const promises = dataset.map((data, i) => {
      const label = labels[data.ys];
      if (!zip.folder(label)) {
        zip.folder(label);
      }
      return createZipEntry(data, label, i);
    });

    // Wait for all promises to complete
    await Promise.all(promises);

    // Generate and save the zip file
    zip.generateAsync({ type: "blob" }).then((content) => {
      console.log("Saved");
      saveAs(content, "dataset.zip");
    });
  };

  const loadDatasetFromZip = (file) => {
    console.log("Function executed");
    const zip = new JSZip();
    zip
      .loadAsync(file)
      .then((contents) => {
        const folders = Object.keys(contents.files).filter(
          (path) => path.endsWith("/") && contents.files[path].dir
        );
        if (folders.length < 2) {
          alert("The zip file must contain at least 2 label folders.");
          return;
        }
        folders.forEach((folder) => {
          const label = folder.slice(0, -1); // Remove the trailing slash
          if (!labels.includes(label)) {
            labels.push(label);
            console.log("Pushed label: " + label);
          }
          Object.keys(contents.files).forEach((filename) => {
            if (filename.startsWith(folder) && filename.endsWith(".png")) {
              contents.files[filename]
                .async("blob")
                .then((blob) => {
                  const img = new Image();
                  img.src = URL.createObjectURL(blob);
                  img.onload = () => {
                    URL.revokeObjectURL(img.src); // Release the object URL
                    const canvas = document.createElement("canvas");
                    canvas.width = 102;
                    canvas.height = 102;
                    const context = canvas.getContext("2d");
                    context.drawImage(img, 0, 0, 102, 102);
                    const imageData = context.getImageData(0, 0, 102, 102);
                    const tensor = tf.browser
                      .fromPixels(imageData, 1)
                      .toFloat()
                      .div(tf.scalar(255));
                    console.log("Added " + filename);
                    setDataset((prevDataset) => [
                      ...prevDataset,
                      { xs: tensor, ys: labels.indexOf(label) },
                    ]);
                  };
                  img.onerror = (err) => {
                    console.error("Image load error: ", err);
                  };
                })
                .catch((err) => console.error("Blob async error: ", err));
            }
          });
        });
      })
      .catch((err) => console.error("Zip load error: ", err));
  };

  const addLabel = () => {
    const labelInput = document.getElementById("labelInput");
    if (labelInput.value !== "") {
      setLabels([...labels, labelInput.value]);
      labelInput.value = "";
    }
  };

  const displayLabelStats = () => {
    const labelStats = labels.reduce((stats, label, i) => {
      const count = dataset.filter((d) => d.ys === i).length;
      stats[label] = count;
      return stats;
    }, {});

    let statsText = "";
    for (const label in labelStats) {
      statsText += `Label ${label}: ${labelStats[label]} samples\n`;
    }

    alert(statsText);
  };

  const displayLabelStatsV = () => {
    const labelStats = labels.reduce((stats, label, i) => {
      const count = dataset.filter((d) => d.ys === i).length;
      stats[label] = count;
      return stats;
    }, {});

    return Object.keys(labelStats).map((label, index) => (
      <div key={index}>
        Label {label}: {labelStats[label]} samples
      </div>
    ));
  };

  const trainModel = async () => {
    setIsTraining(true);
    toast("Model training started...");
    toast("Creating model...");
  
    // Use the selected model to create the new model
    const createModelFunc = modelCreators[selectedModel];
    const newModel = createModelFunc(labels);
  
    toast("Model created");
  
    // Prepare the data
    const xs = tf.concat(dataset.map((d) => d.xs.reshape([1, 102, 102, 1])));
    const ys = tf.oneHot(
      tf.tensor1d(
        dataset.map((d) => d.ys),
        "int32"
      ),
      labels.length
    );
  
    let logsData = [];
    let logstring = "";
  
    await newModel.fit(xs, ys, {
      epochs: epochs,
      batchSize: 8,
      shuffle: true,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          console.log(`Epoch ${epoch + 1} / ${epochs}`);
          console.log(
            `Loss: ${logs.loss.toFixed(4)}, Accuracy: ${logs.acc.toFixed(4)}`
          );
          console.log(
            `Validation Loss: ${logs.val_loss.toFixed(
              4
            )}, Validation Accuracy: ${logs.val_acc.toFixed(4)}`
          );
  
          logsData.push({
            epoch: epoch + 1,
            loss: logs.loss,
            acc: logs.acc,
            val_loss: logs.val_loss,
            val_acc: logs.val_acc,
          });
  
          setTrainLog(
            logstring +
              `Epoch ${epoch + 1} / ${epochs}\nLoss: ${logs.loss.toFixed(
                4
              )}, Accuracy: ${logs.acc.toFixed(
                4
              )}\nValidation Loss: ${logs.val_loss.toFixed(
                4
              )}, Validation Accuracy: ${logs.val_acc.toFixed(
                4
              )}\n--------------------------------------------------\n`
          );
          logstring =
            logstring +
            `Epoch ${epoch + 1} / ${epochs}\nLoss: ${logs.loss.toFixed(
              4
            )}, Accuracy: ${logs.acc.toFixed(
              4
            )}\nValidation Loss: ${logs.val_loss.toFixed(
              4
            )}, Validation Accuracy: ${logs.val_acc.toFixed(
              4
            )}\n--------------------------------------------------\n`;
  
          setTrainingLogs([...logsData]);
  
          // Keep UI responsive
          await tf.nextFrame();
        },
      },
    });
  
    xs.dispose();
    ys.dispose();
  
    setModel(newModel);
    document.getElementById("inferenceContainer").style.display = "block";
    setIsTraining(false);
  };
  

  const clearInferenceCanvas = () => {
    const inferenceCanvas = inferenceCanvasRef.current;
    const inferenceContext = inferenceCanvas.getContext("2d");
    inferenceContext.clearRect(
      0,
      0,
      inferenceCanvas.width,
      inferenceCanvas.height
    );
    inferenceContext.fillStyle = "black";
    inferenceContext.fillRect(
      0,
      0,
      inferenceCanvas.width,
      inferenceCanvas.height
    );
  };

  useEffect(() => {
    const createModelFunc = modelCreators[selectedModel];
    const newModel = createModelFunc();
    setModel(newModel);
  }, [selectedModel]);

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.keyCode === 32) {
        event.preventDefault();
        addDataset();
      }
      if (event.keyCode === 67) {
        clearDrawCanvas();
      }
    };

    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [addDataset, clearDrawCanvas]);

  const clearDatasetLabels = () => {
    setLabels([]);
    setDataset([]);
  };

  return (
    <>
      <div className="App">
        <h1>Model Breeder</h1>
        <div>
          <label style={{ marginRight: "10px" }}>Select Label:</label>
          <input type="text" id="labelInput" placeholder="Enter label name" />
          <button onClick={addLabel}>Add Label</button>
          <select ref={labelSelectRef}>
            {labels.map((label, index) => (
              <option key={index} value={index}>
                {label}
              </option>
            ))}
          </select>
        </div>
        <div
          style={{
            display: "flex",
            flexDirection: "row",
            alignContent: "center",
            justifyContent: "center",
          }}
        >
        <div style={{display: "flex", flexDirection:'column', marginRight:'10px'}}>
        <h2>Select Neural Network</h2>
          <select
            ref={labelSelectRef}
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {modelList.map((label, index) => (
              <option key={index} value={label}>
                {label}
              </option>
            ))}
          </select>
        </div>
          <canvas
            ref={drawCanvasRef}
            width={280}
            height={280}
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseOut={stopDrawing}
            style={{ border: "1px solid black" }}
          />
          <div style={{ display: "flex", flexDirection: "column" }}>
            <button style={{ marginTop: "20px" }} onClick={clearDrawCanvas}>
              Clear Drawing
            </button>
            <button onClick={addDataset}>Add to Dataset</button>
            {dataset.length > 0 && displayLabelStatsV()}
            <input
              type="file"
              ref={(inputElement) => {
                fileInputRef.current = inputElement;
                inputElementRef.current = inputElement;
              }}
              onChange={handleFileChange}
              style={{ display: "none" }}
            />
          </div>
        </div>
        <div>
          <button onClick={trainModel}>Train Model</button>
          <button onClick={displayLabelStats}>See dataset</button>
          {labels.length > 1 && (
            <button style={{ marginTop: "20px" }} onClick={saveDatasetToZip}>
              Save Dataset
            </button>
          )}
          <button
            style={{ marginTop: "20px" }}
            onClick={() => handleButtonClick()}
          >
            Load Dataset
          </button>
          <button
            style={{ marginTOp: "20px" }}
            onClick={() => clearDatasetLabels()}
          >
            Clear Dataset
          </button>
          <button
            onClick={() => (showLog ? setShowLog(false) : setShowLog(true))}
          >
            {showLog ? "Hide Log Output" : "Show Log Output"}
          </button>
        </div>
        <div>
          {showLog && (
            <>
              {" "}
              <textarea
                value={trainLog}
                readOnly
                rows={10}
                cols={60}
                style={{
                  overflow: "auto", // Show both vertical and horizontal scrollbars when needed
                  resize: "vertical", // Allow vertical resizing only
                  minHeight: "100px", // Minimum height (5 rows)
                  maxHeight: "300px", // Maximum height (30 rows)
                  boxSizing: "border-box", // Include padding and border in the height calculation
                  color: "white",
                  backgroundColor: "black",
                }}
              />
            </>
          )}
        </div>
        <hr />
        <div           style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
          }}>
            {isTraining ? (<><h2>Disabled animation while training</h2></>) : (<>        <h1>Model Visualizer</h1>
          {animationPlay && (
            <>
              {" "}
              <ModelVisualizer model={model} animationPlay={animationPlay} />
            </>
          )}

          <button
            onClick={() =>
              animationPlay ? setAnimationPlay(false) : setAnimationPlay(true)
            }
          >
            {animationPlay ? "Stop " : "Start "}Animation
          </button></>)}
          </div>
        <hr></hr>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
          }}
        >

          <TrainingCharts logs={trainingLogs} />
        </div>
        <hr></hr>
        <div
          id="inferenceContainer"
          style={{ display: "none", flexDirection: "column" }}
        >
          <h2>Inference Canvas</h2>
          <canvas
            ref={inferenceCanvasRef}
            width={280}
            height={280}
            onMouseDown={startInferenceDrawing}
            onMouseMove={inferenceDraw}
            onMouseUp={stopInferenceDrawing}
            onMouseOut={stopInferenceDrawing}
            style={{ border: "1px solid black" }}
          />
          <div>
            <h2>
              Predicted Label: <span ref={predictionOutputRef}></span>
            </h2>
            <div ref={predictionProbabilitiesRef}></div>
            <button onClick={clearInferenceCanvas}>Clear Inference</button>
          </div>
        </div>
      </div>
      <ToastContainer
        position="top-right"
        autoClose={700}
        hideProgressBar={false}
        newestOnTop={false}
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme="light"
        transition:Bounce
      />
    </>
  );
}

export default App;
