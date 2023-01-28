import { useRef, useState } from 'react';
import * as React from 'react';
import { inference } from '../utils/predict';
import styles from '../styles/Home.module.css';
import { tensorToImageData, loadImageFromPath } from '../utils/imageHelper';
import { Tensor } from 'onnxruntime-web';

interface Props {
  height: number;
  width: number;
}

const ImageCanvas = (props: Props) => {

  const inputRef = useRef<HTMLInputElement>(null);
  const inputFileRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const canvasOutRef = useRef<HTMLCanvasElement>(null);
  const [resultLabel, setLabel] = useState("");
  const [inferenceTime, setInferenceTime] = useState("");

  // Draw image and other  UI elements then run inference
  const displayImageAndRunInference = () => { 
    // Clear out previous values.
    setLabel(`Inferencing...`);
    setInferenceTime("");
   
    // Run the inference
    submitInference();
  };

  const displayImageData = (data: Uint8ClampedArray, canvas: HTMLCanvasElement) => {
    const ctx = canvas!.getContext('2d');
    var imageData = ctx!.createImageData(props.width, props.height);
    imageData.data.set(data);
    ctx!.putImageData(imageData, 0, 0);
    var dataUri = canvas!.toDataURL();
  }

  const submitInference = async () => {
    // Get the image data from the canvas and submit inference.
    var inputImage = await loadImageFromPath(inputRef.current!.value);
    inputImage.bitmap.data = inputImage.bitmap.data.slice(0, props.width * props.height * 4);
    displayImageData(Uint8ClampedArray.from(inputImage.bitmap.data), canvasRef.current!);

    var [resultTensor, inferenceTime] = await inference(inputImage);

    displayImageData(tensorToImageData(resultTensor), canvasOutRef.current!);

    // Update the label and confidence
    setLabel("");
    setInferenceTime(`Inference time: ${inferenceTime} seconds`);
  };

  const selectFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    var URL = window.webkitURL || window.URL;
    var url = URL.createObjectURL(e.target.files![0]);
    inputRef.current!.value = url;
  }

  const enterURL = (e: React.ChangeEvent<HTMLInputElement>) => {
    inputFileRef.current!.value = "";
  }

  return (
    <>
      <input className={styles.grid} type="file" ref={inputFileRef} onChange={selectFile}/>
      <div className={styles.grid}>
        <label>URL: </label>
        <input type="text" ref={inputRef} className={styles.textInput} onChange={enterURL}/>
      </div>
      <button
        className={styles.grid}
        onClick={displayImageAndRunInference} >
        Run inference
      </button>
      <br/>
      <div className={styles.canvasContainer}>
        <canvas className={styles.cv} ref={canvasRef} width={props.width} height={props.height} />
        <canvas className={styles.cv} ref={canvasOutRef} width={props.width} height={props.height} />
      </div>
      <span>{resultLabel}</span>
      <span>{inferenceTime}</span>
    </>
  )

};

export default ImageCanvas;
