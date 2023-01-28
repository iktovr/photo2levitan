import * as Jimp from 'jimp';
import { Tensor } from 'onnxruntime-web';

export async function loadImageFromPath(path: string): Promise<Jimp> {
  // Use Jimp to load the image and resize it.
  var imageData = await Jimp.default.read(path).then((image: Jimp) => {
    var newWidth: number, newHeight: number;
    if (image.bitmap.width < image.bitmap.height) {
      newWidth = 256;
      newHeight = 256 * image.bitmap.height / image.bitmap.width;
    } else {
      newHeight = 256;
      newWidth = 256 * image.bitmap.width / image.bitmap.height;
    }

    var cropX: number, cropY: number;
    if (newWidth < newHeight) {
      cropX = 0;
      cropY = newHeight / 2 - 128;
    } else if (newHeight < newWidth) {
      cropY = 0;
      cropX = newWidth / 2 - 128;
    } else {
      cropX = 0;
      cropY = 0;
    }
    return image.resize(newWidth, newHeight).crop(cropX, cropY, 256, 256);
  });

  return imageData;
}

export function imageDataToTensor(image: Jimp, dims: number[] = [1, 3, 256, 256]): Tensor {
  // 1. Get buffer data from image and create R, G, and B arrays.
  var imageBufferData = image.bitmap.data;
  const [redArray, greenArray, blueArray] = new Array(new Array<number>(), new Array<number>(), new Array<number>());

  // 2. Loop through the image buffer and extract the R, G, and B channels
  for (let i = 0; i < imageBufferData.length; i += 4) {
    redArray.push(imageBufferData[i]);
    greenArray.push(imageBufferData[i + 1]);
    blueArray.push(imageBufferData[i + 2]);
    // skip data[i + 3] to filter out the alpha channel
  }

  // 3. Concatenate RGB to transpose [256, 256, 3] -> [3, 256, 256] to a number array
  const transposedData = redArray.concat(greenArray).concat(blueArray);

  // 4. convert to float32
  let i, l = transposedData.length; // length, we need this for the loop
  // create the Float32Array size 3 * 256 * 256 for these dimensions output
  const float32Data = new Float32Array(dims[1] * dims[2] * dims[3]);
  for (i = 0; i < l; i++) {
    float32Data[i] = transposedData[i] / 255.0 * 2 - 1; // convert to float
  }
  // 5. create the tensor object from onnxruntime-web.
  const inputTensor = new Tensor("float32", float32Data, dims);
  return inputTensor;
}

export function tensorToImageData(tensor: Tensor, width: number = 256, height: number = 256): Uint8ClampedArray {
  var data = new Uint8ClampedArray(width * height * 4);

  for (var i = 0; i < width * height; ++i) {
    data[i * 4    ] = (Number(tensor.data[i]) / 2.0 + 0.5) * 255;
    data[i * 4 + 1] = (Number(tensor.data[width * height + i]) / 2.0 + 0.5) * 255;
    data[i * 4 + 2] = (Number(tensor.data[2 * width * height + i]) / 2.0 + 0.5) * 255;
    data[i * 4 + 3] = 255;
  }

  return data;
}
