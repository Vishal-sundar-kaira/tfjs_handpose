import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, Platform, SafeAreaView, Button } from 'react-native';
import { Camera } from 'expo-camera';
import * as tf from "@tensorflow/tfjs";
import { cameraWithTensors } from '@tensorflow/tfjs-react-native';
import * as handpose from '@tensorflow-models/handpose';
import * as Speech from 'expo-speech'
import { Dimensions } from 'react-native';
import Canvas from "react-native-canvas";

const { width, height } = Dimensions.get("window");

console.log('Screen Height:', height);
console.log('Screen Width:', width);

const MODEL_CONFIG = {
  detectionConfidence: 0.8,
  maxContinuousChecks: 10,
  maxNumBoxes: 20, // maximum boxes to detect
  iouThreshold: 0.5, // ioU thresho non-max suppression
  scoreThreshold: 0.8, // confidence
};

// start

const RESIZE_WIDTH = 224;
const RESIZE_HEIGHT = Math.round((height / width) * RESIZE_WIDTH);

const FPS = 15; // set desired FPS for predictions

// Hand part connections
const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4], // Thumb
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8], // Index
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12], // Middle
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16], // Ring
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20], // Pinky
];

const textureDims = Platform.OS === 'ios' ?
  {
    height: 1920,
    width: 1080,
  } :
  {
    height: 1200,
    width: 1600,
  };

const TensorCamera = cameraWithTensors(Camera);

export default function App() {
  const [hasPermission, setHasPermission] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [handposeModel, setHandposeModel] = useState();
  const [handposeModelLoaded, setHandposeModelLoaded] = useState(false);

  const [handmodel, sethandmodel] = useState(null);
  let context = useRef();
  let canvas = useRef();
  const [ismodelready, setismodelready] = useState(false);

  const class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'nothing', 'space', 'del'];

  const cameraActiveRef = useRef(false);

  const initialiseTensorflow = async () => {
    await tf.ready();
    tf.getBackend();
    console.log("TensorFlow is ready");
  }

function drawKeypoints(keypoints, ctx) {
    keypoints.forEach((keypoint) => {
        ctx.beginPath();
        ctx.arc(keypoint[0], keypoint[1], 5, 0, 2 * Math.PI);
        ctx.fillStyle = "red";
        ctx.fill();
    });
}

function drawConnections(pairs, ctx) {
  console.log("drawconnections started")
    ctx.beginPath();
    ctx.strokeStyle = "blue";
    ctx.lineWidth = 2;
    console.log("pairs started")
    // console.log(pairs)
    // console.log(Array.isArray(pairs))
    pairs.forEach(([start, end]) => {
      console.log(start[0],start[1])
        ctx.moveTo(start[0], start[1]);
        ctx.lineTo(end[0], end[1]);
    });
    console.log("pairs end")
    ctx.stroke();
}

function mapPoints(predictions, nextImageTensor) {
  if (!context.current || !canvas.current) {
      console.log("no context or canvas");
      return;
  }
  console.log("mapped point started")
  console.log("mapped2")
  const scaleWidth = width / nextImageTensor.shape[1];
  console.log("mapped problem in nextImagetesnor")
  const scaleHeight = height / nextImageTensor.shape[0];

  const flipHorizontal = false; // Set flipHorizontal to false

  context.current.clearRect(0, 0, width, height);
  console.log("prediction loop started")
  for (const prediction of predictions) {
      const keypoints = prediction.landmarks.map((landmark) => {
          const x = flipHorizontal
              ? canvas.current.width - landmark[0] * scaleWidth
              : landmark[0] * scaleWidth;
          const y = landmark[1] * scaleHeight;
          return [x, y];
      });
      console.log("drawkeypoints")
      drawKeypoints(keypoints, context.current);
      console.log("keypoint done ")
      drawConnections(
          HAND_CONNECTIONS.map(([startIdx, endIdx]) => [
              keypoints[startIdx],
              keypoints[endIdx],
          ]),
          context.current
      );
      console.log("connection drawn ")
  }
}

const handleCanvas = async (can) => {
  if (can) {
    console.log("canvas set")
      can.width = width;
      can.height = height;
      const ctx = can.getContext("2d");
      context.current = ctx;
      ctx.strokeStyle = "red";
      ctx.fillStyle = "red";
      ctx.lineWidth = 3;
      canvas.current = can;
  }
};

let textureDims;
Platform.OS === "ios"
  ? (textureDims = { height: 1920, width: 1080 })
  : (textureDims = { height: 1200, width: 1600 });



  const handleCameraStream = (images) => {
    const loop = async () => {
      if (!handposeModel) {
        console.log("Handpose model not loaded yet");
        return;
      }
      const nextImageTensor = images.next().value;
      if (nextImageTensor && cameraActiveRef.current) {
        try {
          const predictions=await handposeModel.estimateHands(nextImageTensor)
          console.log(predictions)
          if(predictions&&predictions.length>0){
            console.log("mappoint is starting")
            mapPoints(predictions,nextImageTensor)
            tf.dispose(nextImageTensor);
          }
        }
        catch (error) {
          console.error("Error predicting:", error);
        } 
      }
      setTimeout(loop,1000 / FPS);
    };
    loop();
  };

  
  useEffect(() => {
    (async () => {
      try {
        const { status } = await Camera.requestCameraPermissionsAsync();
        setHasPermission(status === 'granted');
        await initialiseTensorflow();
        console.log("Loading model...");
        const modelJson = require("./assets/models4/model.json");
        const model = await tf.loadLayersModel(tf.io.fromMemory(modelJson));
        setModel(model);
        console.log("Model loaded successfully");
        const model2 = await handpose.load(MODEL_CONFIG);
        setHandposeModel(model2);
        setHandposeModelLoaded(true);
        console.log("Handpose model loaded successfully");
      } catch (error) {
        console.error("Error loading model:", error);
      }
    })();
  }, []);

  if (hasPermission === null) {
    return <View />;
  }
  if (hasPermission === false) {
    return <Text>No access to camera</Text>;
  }

  if (!handposeModelLoaded) {
    return <Text>Loading handpose model...</Text>;
  }

  const startCamera = () => {
    cameraActiveRef.current = true;
  };

  const stopCamera = () => {
    cameraActiveRef.current = false;
    Speech.speak(predictions.join(''))
    setPredictions([]);
    setCurr();
  };

  return (
    <SafeAreaView style={styles.container}>
       <TensorCamera
        // Standard Camera props
        style={styles.camera}
        type={Camera.Constants.Type.back}
        // Tensor related props
        cameraTextureHeight={textureDims.height}
        cameraTextureWidth={textureDims.width}
        resizeHeight={RESIZE_HEIGHT}
        resizeWidth={RESIZE_WIDTH}
        resizeDepth={3}
        onReady={handleCameraStream}
        autorender={true}
        useCustomShadersToResize={false}
      />

      <Canvas style={styles.canvas} ref={handleCanvas} />
      <View style={styles.buttonContainer}>
        <Button title="Start Prediction" onPress={startCamera} />
        <Button title="Stop Prediction" onPress={stopCamera} />
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
  },
  landmark: {
    position: 'absolute',
    width: 5,
    height: 5,
    borderRadius: 5 / 2,
    backgroundColor: 'red',
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 10,
  },
  camera: {
    width: "100%",
    height: "100%",
  },
  canvas: {
    position: "absolute",
    zIndex: 1000000,
    width: width, // Use the numeric value directly
    height: height, // Use the numeric value directly
  },
});