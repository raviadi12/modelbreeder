import * as tf from "@tensorflow/tfjs";
import * as mobilenet from 'tensorflow-models/mobilenet';

const createMobileNetModel = async (labels) => {
  // Load the pre-trained MobileNet model
  const loadedModel = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // Print the names of the layers to understand the architecture
  loadedModel.layers.forEach(layer => console.log(layer.name));

  // Select the layer before the final classification layers
  const layer = loadedModel.getLayer('conv_pw_13_relu');
  const truncatedModel = tf.model({
    inputs: loadedModel.inputs,
    outputs: layer.output
  });

  // Freeze the layers of the truncated MobileNet model
  for (let l of truncatedModel.layers) {
    l.trainable = false;
  }

  // Create a new sequential model
  const model = tf.sequential();
  model.add(tf.layers.inputLayer({ inputShape: [102, 102, 1] }));

  // Add a convolutional layer to simulate the RGB channels from grayscale
  model.add(tf.layers.conv2d({
    filters: 3,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu',
  }));

  model.add(truncatedModel);

  // Add custom layers on top of the truncated MobileNet model
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
  model.add(tf.layers.dense({ units: labels ? labels.length : 10, activation: 'softmax' }));

  // Compile the model
  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
};


const createAlexNet = () => {
    const alexNet = tf.sequential();
  
    // First Convolutional Layer
    alexNet.add(
      tf.layers.conv2d({
        inputShape: [227, 227, 3], // Input shape: 227x227 RGB images
        filters: 96,
        kernelSize: 11,
        strides: 4,
        activation: "relu",
        kernelInitializer: "heNormal",
      })
    );
    alexNet.add(tf.layers.maxPooling2d({ poolSize: 3, strides: 2 }));
    alexNet.add(tf.layers.batchNormalization());
  
    // Second Convolutional Layer
    alexNet.add(
      tf.layers.conv2d({
        filters: 256,
        kernelSize: 5,
        padding: "same",
        activation: "relu",
        kernelInitializer: "heNormal",
      })
    );
    alexNet.add(tf.layers.maxPooling2d({ poolSize: 3, strides: 2 }));
    alexNet.add(tf.layers.batchNormalization());
  
    // Third Convolutional Layer
    alexNet.add(
      tf.layers.conv2d({
        filters: 384,
        kernelSize: 3,
        padding: "same",
        activation: "relu",
        kernelInitializer: "heNormal",
      })
    );
  
    // Fourth Convolutional Layer
    alexNet.add(
      tf.layers.conv2d({
        filters: 384,
        kernelSize: 3,
        padding: "same",
        activation: "relu",
        kernelInitializer: "heNormal",
      })
    );
  
    // Fifth Convolutional Layer
    alexNet.add(
      tf.layers.conv2d({
        filters: 256,
        kernelSize: 3,
        padding: "same",
        activation: "relu",
        kernelInitializer: "heNormal",
      })
    );
    alexNet.add(tf.layers.maxPooling2d({ poolSize: 3, strides: 2 }));
    alexNet.add(tf.layers.batchNormalization());
  
    // Flatten and Fully Connected Layers
    alexNet.add(tf.layers.flatten());
    alexNet.add(
      tf.layers.dense({
        units: 4096,
        activation: "relu",
        kernelInitializer: "heNormal",
      })
    );
    alexNet.add(tf.layers.dropout(0.5));
    alexNet.add(
      tf.layers.dense({
        units: 4096,
        activation: "relu",
        kernelInitializer: "heNormal",
      })
    );
    alexNet.add(tf.layers.dropout(0.5));
    alexNet.add(
      tf.layers.dense({
        units: 1000, // Replace with the desired number of output classes
        activation: "softmax",
        kernelInitializer: "glorotNormal",
      })
    );
  
    return alexNet;
  };

  const createModel = (labels) => {
    const model = tf.sequential();
    model.add(
      tf.layers.conv2d({
        inputShape: [102, 102, 1],
        filters: 32,
        kernelSize: 3,
        activation: "relu",
      })
    );
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(
      tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: "relu" })
    );
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(
      tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: "relu" })
    );
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: labels ? labels.length : 10, activation: "softmax" }));
    model.compile({
      optimizer: "adam",
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });
    return model;
  };

  const createComplexModel = (labels) => {
    const model = tf.sequential();
    
    // First convolutional block
    model.add(
      tf.layers.conv2d({
        inputShape: [102, 102, 1],
        filters: 32,
        kernelSize: 3,
        activation: "relu",
      })
    );
    model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: "relu" }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.dropout({ rate: 0.25 }));
    
    // Second convolutional block
    model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: "relu" }));
    model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: "relu" }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.dropout({ rate: 0.25 }));
    
    // Third convolutional block
    model.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: "relu" }));
    model.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: "relu" }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.dropout({ rate: 0.25 }));
    
    // Fourth convolutional block
    model.add(tf.layers.conv2d({ filters: 256, kernelSize: 3, activation: "relu" }));
    model.add(tf.layers.conv2d({ filters: 256, kernelSize: 3, activation: "relu" }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.dropout({ rate: 0.25 }));
    
    // Fully connected layers
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dropout({ rate: 0.5 }));
    model.add(tf.layers.dense({ units: labels ? labels.length : 10, activation: "softmax" }));
    
    model.compile({
      optimizer: tf.train.adam(),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });
    
    return model;
  };
  


  const createSimpleModel = (labels) => {
    const model = tf.sequential();
    model.add(
      tf.layers.conv2d({
        inputShape: [102, 102, 1],
        filters: 16,
        kernelSize: 3,
        activation: "relu",
      })
    );
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(
      tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: "relu" })
    );
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(
      tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: "relu" })
    );
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 128, activation: "relu" }));
    model.add(tf.layers.dense({ units: labels ? labels.length : 10, activation: "softmax" }));
    model.compile({
      optimizer: "adam",
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });
    return model;
  };
  
  

  const createDecoyModel = () => {
    const decoyModel = tf.sequential();
    decoyModel.add(
      tf.layers.conv2d({
        inputShape: [102, 102, 1],
        filters: 32,
        kernelSize: 3,
        activation: "relu",
      })
    );
    decoyModel.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    decoyModel.add(
      tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: "relu" })
    );
    decoyModel.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    decoyModel.add(
      tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: "relu" })
    );
    decoyModel.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    decoyModel.add(tf.layers.flatten());
    decoyModel.add(tf.layers.dense({ units: 256, activation: "relu" }));
    decoyModel.add(tf.layers.dense({ units: 10, activation: "softmax" }));
    return decoyModel;
  };


export { createAlexNet, createModel, createDecoyModel, createMobileNetModel, createSimpleModel, createComplexModel }