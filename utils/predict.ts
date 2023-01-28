// Language: typescript
// Path: react-next\utils\predict.ts
import { imageDataToTensor } from './imageHelper';
import { runModel } from './modelHelper';
import { Tensor } from 'onnxruntime-web';
import * as Jimp from 'jimp';

export async function inference(image: Jimp): Promise<[Tensor,number]> {
  // 1. Convert image to tensor
  const imageTensor = imageDataToTensor(image);
  // 2. Run model
  const [result, inferenceTime] = await runModel(imageTensor);
  // 3. Return predictions and the amount of time it took to inference.
  return [result, inferenceTime];
}

