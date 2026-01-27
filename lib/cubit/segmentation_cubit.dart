import 'dart:async';
import 'dart:io';
import 'dart:ui' as ui;
import 'package:bloc/bloc.dart';
import 'package:camera/camera.dart';
import 'dart:developer' as developer;
import 'package:equatable/equatable.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:integral_isolates/integral_isolates.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import '../segmentation_isolate.dart';

part 'segmentation_state.dart';

class SegmentationCubit extends Cubit<SegmentationState> {
  SegmentationCubit() : super(const SegmentationState());

  late final StatefulIsolate _isolate;
  bool _isProcessing = false;
  bool _isModelsLoaded = false;

  Future<void> initialize() async {
    emit(state.copyWith(status: SegmentationStatus.loadingModel));
    _isolate = StatefulIsolate();

    try {
      final directory = await getApplicationDocumentsDirectory();

      // Prepare Feature Extractor (DINOv3)
      const featureModelName = 'dinov3_feature_extractor.onnx';
      final featureBytes = await rootBundle.load('assets/$featureModelName');
      final featurePath = p.join(directory.path, featureModelName);
      await File(featurePath).writeAsBytes(featureBytes.buffer.asUint8List());

      // Try to find a saved custom model
      // We look for a file we saved previously, e.g., 'current_classifier.onnx'
      final savedClassifierPath = p.join(
        directory.path,
        'current_classifier.onnx',
      );
      final hasSavedModel = await File(savedClassifierPath).exists();

      final token = RootIsolateToken.instance!;

      // Initialize Isolate
      // We pass null for classifierModelPath if no file exists
      await _isolate.compute(initializeSession, {
        'featureModelPath': featurePath,
        'classifierModelPath': hasSavedModel ? savedClassifierPath : null,
        'token': token,
      });

      _isModelsLoaded = true;

      emit(
        state.copyWith(
          status: SegmentationStatus.modelReady,
          // If we didn't load a classifier, we are "Ready" but cannot segment yet
          errorMessage: hasSavedModel
              ? null
              : 'No model selected. Please manage models.',
        ),
      );
    } catch (e) {
      emit(
        state.copyWith(
          status: SegmentationStatus.error,
          errorMessage: 'Failed to initialize: $e',
        ),
      );
    }
  }

  void _logError(String message, Object error, [StackTrace? stackTrace]) {
    // Structured logging to the DevTools console
    developer.log(
      message,
      name: 'SegmentationCubit',
      error: error,
      stackTrace: stackTrace, // Captures the full path to the bug
    );

    emit(
      state.copyWith(
        status: SegmentationStatus.error,
        errorMessage: '$message: $error',
      ),
    );
  }

  void processCameraImage(CameraImage cameraImage) {
    // Basic checks: Model must be ready, not currently processing, and segmentation enabled
    if (!state.isSegmenting || _isProcessing || !_isModelsLoaded) {
      return;
    }

    _isProcessing = true;

    _isolate
        .compute(runSegmentation, {
          'planes': cameraImage.planes.map((p) => p.bytes).toList(),
          'width': cameraImage.width,
          'height': cameraImage.height,
          'format': cameraImage.format.group,
          'threshold': state.similarityThreshold,
          'inputSize': state.selectedInputSize,
          'showLargestOnly': state.showLargestAreaOnly,
        })
        .then((result) {
          if (state.isSegmenting && result.isNotEmpty) {
            _updateOverlay(
              result['scores'] as List<double>,
              result['width'] as int,
              result['height'] as int,
            );
          }
          _isProcessing = false;
        });
  }

  void _updateOverlay(List<double> scores, int width, int height) {
    final pixels = Uint8List(width * height * 4);
    for (int i = 0; i < scores.length; i++) {
      // Logic: If probability > threshold, draw pixel
      if (scores[i] > state.similarityThreshold) {
        pixels[i * 4 + 0] = 30; // R
        pixels[i * 4 + 1] = 255; // G
        pixels[i * 4 + 2] = 150; // B
        pixels[i * 4 + 3] = 170; // A
      }
    }

    ui.decodeImageFromPixels(pixels, width, height, ui.PixelFormat.rgba8888, (
      img,
    ) {
      emit(state.copyWith(overlayImage: img));
    });
  }

  void toggleSegmentation() {
    // If models aren't loaded, don't start
    if (state.status != SegmentationStatus.modelReady) return;

    final newIsSegmenting = !state.isSegmenting;
    emit(
      state.copyWith(
        isSegmenting: newIsSegmenting,
        clearOverlay: !newIsSegmenting,
      ),
    );
  }

  void updateInputSize(int newSize) {
    if (newSize != state.selectedInputSize) {
      emit(
        state.copyWith(
          selectedInputSize: newSize,
          isSegmenting: false,
          status: SegmentationStatus.modelReady,
          clearOverlay: true,
        ),
      );
    }
  }

  void updateThreshold(double value) =>
      emit(state.copyWith(similarityThreshold: value));

  void toggleLargestAreaOnly() =>
      emit(state.copyWith(showLargestAreaOnly: !state.showLargestAreaOnly));

  void toggleSliderVisibility() =>
      emit(state.copyWith(showSlider: !state.showSlider));

  Future<void> loadCustomClassifier(String newModelPath) async {
    try {
      emit(state.copyWith(status: SegmentationStatus.loadingModel));

      // 1. Save this as the "current" default for next app restart
      final directory = await getApplicationDocumentsDirectory();
      final persistentPath = p.join(directory.path, 'current_classifier.onnx');
      await File(newModelPath).copy(persistentPath);

      // 2. Hot-swap
      final token = RootIsolateToken.instance!;
      await _isolate.compute(reloadClassifier, {
        'path': persistentPath,
        'token': token,
      });

      emit(
        state.copyWith(
          status: SegmentationStatus.modelReady,
          similarityThreshold: 0.7, // Reset threshold
          errorMessage: null, // Clear any "No model" warning
        ),
      );
    } catch (e, stackTrace) {
      _logError('Failed to load new model', e, stackTrace);
    }
  }

  @override
  Future<void> close() {
    _isolate.dispose();
    return super.close();
  }
}
