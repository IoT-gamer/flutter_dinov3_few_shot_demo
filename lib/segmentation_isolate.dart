import 'dart:io';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'dart:developer' as developer;
import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

const int patchSize = 16;
final imagenetMean = [0.485, 0.456, 0.406];
final imagenetStd = [0.229, 0.224, 0.225];

// We now hold TWO distinct sessions
OrtSession? _featureSession; // The heavy DINOv3 (Static)
OrtSession? _classifierSession; // The light Logistic Regression (Dynamic)

Future<void> initializeSession(Map<String, dynamic> args) async {
  BackgroundIsolateBinaryMessenger.ensureInitialized(
    args['token'] as RootIsolateToken,
  );

  final ort = OnnxRuntime();
  // Standard provider setup
  final providers = Platform.isAndroid
      ? [OrtProvider.NNAPI, OrtProvider.CPU]
      : Platform.isIOS
      ? [OrtProvider.CORE_ML, OrtProvider.CPU]
      : [OrtProvider.CPU];
  final options = OrtSessionOptions(providers: providers);

  // Initialize Feature Extractor (Always loaded)
  final String featureModelPath = args['featureModelPath'];
  _featureSession = await ort.createSession(featureModelPath, options: options);
  print('✅ DINOv3 Feature Extractor Session Initialized.');

  // Initialize Classifier (Optional/Dynamic)
  // If we are starting with a pre-trained asset, load it here.
  if (args['classifierModelPath'] != null) {
    final String classifierPath = args['classifierModelPath'];
    _classifierSession = await ort.createSession(
      classifierPath,
      options: options,
    );
    print('✅ Classifier Session Initialized.');
  }
}

/// Runs the split-model inference pipeline:
/// Camera Frame -> DINOv3 (Features) -> Logistic Regression (Probabilities) -> Mask
Future<Map<String, dynamic>> runSegmentation(Map<String, dynamic> args) async {
  // Safety Check: Ensure both models are loaded
  if (_featureSession == null || _classifierSession == null) {
    print('⚠️ Segmentation skipped: Models not fully initialized.');
    return {};
  }

  // Unpack arguments
  final List<Uint8List> planes = args['planes'];
  final ImageFormatGroup format = args['format'] ?? ImageFormatGroup.yuv420;
  final int width = args['width'];
  final int height = args['height'];
  final double similarityThreshold = args['threshold'] ?? 0.5;
  final int imageSize = args['inputSize'];
  final bool showLargestOnly = args['showLargestOnly'] ?? false;

  // OpenCV Mats (manual memory management required)
  cv.Mat? yuvMat, rgbMat, rotatedMat, resizedMat;
  // Post-processing Mats
  cv.Mat? maskMat, labels, stats, centroids;

  // Tensors (must be disposed)
  OrtValue? inputTensor;
  OrtValue? classifierInputTensor;

  // Session Outputs (must be disposed)
  Map<String, OrtValue>? featureOutputs;
  Map<String, OrtValue>? classifierOutputs;

  try {
    // -----------------------------------------------------------
    // STEP 1: Image Pre-processing (YUV -> RGB -> Resize -> Norm)
    // -----------------------------------------------------------
    if (format == ImageFormatGroup.bgra8888) {
      final bgraMat = cv.Mat.fromList(
        height,
        width,
        cv.MatType.CV_8UC4,
        planes[0],
      );
      rgbMat = cv.cvtColor(bgraMat, cv.COLOR_BGRA2RGB);
      bgraMat.dispose();
    } else if (format == ImageFormatGroup.yuv420) {
      final int yuvSize = width * height * 3 ~/ 2;
      final yuvBytes = Uint8List(yuvSize);
      yuvBytes.setRange(0, width * height, planes[0]);
      yuvBytes.setRange(width * height, width * height * 5 ~/ 4, planes[1]);
      yuvBytes.setRange(width * height * 5 ~/ 4, yuvSize, planes[2]);

      yuvMat = cv.Mat.fromList(
        height * 3 ~/ 2,
        width,
        cv.MatType.CV_8UC1,
        yuvBytes,
      );
      rgbMat = cv.cvtColor(yuvMat, cv.COLOR_YUV2RGB_I420);
    } else {
      return {};
    }

    rotatedMat = cv.rotate(rgbMat, cv.ROTATE_90_CLOCKWISE);

    // Calculate Patch-Aligned Dimensions
    final int hPatches = imageSize ~/ patchSize;
    int wPatches =
        (rotatedMat.cols * imageSize) ~/ (rotatedMat.rows * patchSize);
    if (wPatches % 2 != 0) wPatches -= 1;

    final int newH = hPatches * patchSize;
    final int newW = wPatches * patchSize;

    resizedMat = cv.resize(rotatedMat, (
      newW,
      newH,
    ), interpolation: cv.INTER_CUBIC);

    // Normalize to Float32List (ImageNet stats)
    final preprocessed = _preprocessImageFromBytes(resizedMat.data, newW, newH);

    inputTensor = await OrtValue.fromList(
      preprocessed['input_tensor'] as Float32List,
      preprocessed['shape'] as List<int>,
    );

    // -----------------------------------------------------------
    // STEP 2: DINOv3 Feature Extraction
    // -----------------------------------------------------------
    final featureInputs = {'input_image': inputTensor};

    // Run DINOv3
    // Returns Map<String, OrtValue>
    featureOutputs = await _featureSession!.run(featureInputs);

    // FIXED: Access via keys or values.first
    final featuresTensor = featureOutputs!.values.first;

    // Optimization: Reuse buffer
    final List<dynamic> flatFeatures = await featuresTensor.asFlattenedList();
    final floatFeatures = Float32List.fromList(flatFeatures.cast<double>());

    // Determine dimensions
    final int numPatches = preprocessed['num_patches'];
    final int featureDim = floatFeatures.length ~/ numPatches;

    // Create Input Tensor for Classifier
    classifierInputTensor = await OrtValue.fromList(floatFeatures, [
      numPatches,
      featureDim,
    ]);

    // -----------------------------------------------------------
    // STEP 3: Logistic Regression Classifier
    // -----------------------------------------------------------
    final classifierInputs = {'patch_features': classifierInputTensor};

    // Run Classifier
    // Returns Map<String, OrtValue>
    classifierOutputs = await _classifierSession!.run(classifierInputs);

    // skl2onnx output 0 is 'label' (integers)
    // skl2onnx output 1 is 'probabilities' (float tensor or sequence of maps)
    final probTensor = classifierOutputs.values.elementAt(1);
    final List<dynamic> flatProbs = await probTensor.asFlattenedList();

    List<double> finalScores = [];
    // Every even index (0, 2, 4) is Background
    // Every odd index (1, 3, 5) is Foreground
    for (int i = 1; i < flatProbs.length; i += 2) {
      finalScores.add(flatProbs[i] as double);
    }

    // -----------------------------------------------------------
    // STEP 4: Post-Processing (Largest Area Filter)
    // -----------------------------------------------------------
    if (showLargestOnly) {
      final maskData = Uint8List.fromList(
        finalScores.map((s) => s > similarityThreshold ? 255 : 0).toList(),
      );

      maskMat = cv.Mat.fromList(
        hPatches,
        wPatches,
        cv.MatType.CV_8UC1,
        maskData,
      );

      labels = cv.Mat.empty();
      stats = cv.Mat.empty();
      centroids = cv.Mat.empty();

      cv.connectedComponentsWithStats(
        maskMat,
        labels,
        stats,
        centroids,
        8,
        cv.MatType.CV_32S,
        cv.CCL_DEFAULT,
      );

      if (stats.rows > 1) {
        int maxArea = 0;
        int largestComponentLabel = 0;

        for (int i = 1; i < stats.rows; i++) {
          final area = stats.at<int>(i, cv.CC_STAT_AREA);
          if (area > maxArea) {
            maxArea = area;
            largestComponentLabel = i;
          }
        }

        if (largestComponentLabel != 0) {
          final filteredScores = List<double>.filled(finalScores.length, 0.0);
          final labelsData = labels.data.buffer.asInt32List();

          for (int i = 0; i < finalScores.length; i++) {
            if (labelsData[i] == largestComponentLabel) {
              filteredScores[i] = finalScores[i];
            }
          }
          finalScores = filteredScores;
        }
      }
    }

    print(
      "Isolate Stats: Scores=${finalScores.length}, W=$wPatches, H=$hPatches",
    );
    print(
      "Top Score: ${finalScores.reduce((a, b) => a > b ? a : b)}",
    ); // Check if > threshold
    return {
      'scores': finalScores,
      'width': preprocessed['w_patches'],
      'height': preprocessed['h_patches'],
    };
  } catch (e) {
    print("❌ Error inside isolate: $e");
    return {};
  } finally {
    // -----------------------------------------------------------
    // Memory Cleanup
    // -----------------------------------------------------------
    inputTensor?.dispose();
    classifierInputTensor?.dispose();

    // Iterate over Map values to dispose
    featureOutputs?.values.forEach((element) => element.dispose());
    classifierOutputs?.values.forEach((element) => element.dispose());

    yuvMat?.dispose();
    rgbMat?.dispose();
    rotatedMat?.dispose();
    resizedMat?.dispose();
    maskMat?.dispose();
    labels?.dispose();
    stats?.dispose();
    centroids?.dispose();
  }
}

/// Helper: Converts RGB bytes to Normalized Float32List (NCHW format)
Map<String, dynamic> _preprocessImageFromBytes(
  Uint8List imageBytes,
  int newW,
  int newH,
) {
  final int hPatches = newH ~/ patchSize;
  final int wPatches = newW ~/ patchSize;
  final int numPatches = wPatches * hPatches;

  // Create output buffer
  final inputTensor = Float32List(1 * 3 * newH * newW);
  int bufferIndex = 0;

  // Rearrange RGB [H, W, 3] -> NCHW [1, 3, H, W] and Normalize
  for (int c = 0; c < 3; c++) {
    for (int y = 0; y < newH; y++) {
      for (int x = 0; x < newW; x++) {
        final int pixelIndex = (y * newW + x) * 3;
        final double val = imageBytes[pixelIndex + c] / 255.0;
        inputTensor[bufferIndex++] = (val - imagenetMean[c]) / imagenetStd[c];
      }
    }
  }

  return {
    'input_tensor': inputTensor,
    'shape': [1, 3, newH, newW],
    'w_patches': wPatches,
    'h_patches': hPatches,
    'num_patches': numPatches,
  };
}

/// Hot-swaps the classifier session without touching the DINOv3 session.
Future<void> reloadClassifier(Map<String, dynamic> args) async {
  try {
    BackgroundIsolateBinaryMessenger.ensureInitialized(
      args['token'] as RootIsolateToken,
    );
    final String classifierPath = args['path'];
    final ort = OnnxRuntime();

    // Use the same providers/options as before
    final providers = Platform.isAndroid
        ? [OrtProvider.NNAPI, OrtProvider.CPU]
        : Platform.isIOS
        ? [OrtProvider.CORE_ML, OrtProvider.CPU]
        : [OrtProvider.CPU];
    final options = OrtSessionOptions(providers: providers);

    // Close the old session if it exists to free memory
    if (_classifierSession != null) {
      try {
        _classifierSession!.close();
      } catch (e) {
        print('⚠️ Error closing old classifier: $e');
      }
    }

    // Load the new one
    _classifierSession = await ort.createSession(
      classifierPath,
      options: options,
    );
    print('✅ Classifier Hot-Swapped: $classifierPath');
    developer.log('✅ Classifier Hot-Swapped', name: 'SegmentationIsolate');
  } catch (e, stackTrace) {
    // Print to standard error so it shows up even if the isolate crashes
    stderr.writeln('❌ Isolate Hot-Swap Error: $e');
    stderr.writeln(stackTrace);
    rethrow; // Ensure the Cubit catch block receives the error
  }
}
