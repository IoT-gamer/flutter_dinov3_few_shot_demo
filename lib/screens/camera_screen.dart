import 'dart:ui' as ui;
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import '../cubit/segmentation_cubit.dart';
import 'model_manager_screen.dart';

class CameraScreen extends StatelessWidget {
  const CameraScreen({super.key, required this.camera});
  final CameraDescription camera;

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (_) => SegmentationCubit()..initialize(),
      child: CameraView(camera: camera),
    );
  }
}

class CameraView extends StatefulWidget {
  const CameraView({super.key, required this.camera});
  final CameraDescription camera;

  @override
  State<CameraView> createState() => _CameraViewState();
}

class _CameraViewState extends State<CameraView> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  int _frameCounter = 0;

  @override
  void initState() {
    super.initState();
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.high,
      enableAudio: false,
    );
    _initializeControllerFuture = _controller.initialize().then((_) {
      _controller.startImageStream(_processCameraImage);
    });
  }

  void _processCameraImage(CameraImage cameraImage) {
    _frameCounter++;
    if (_frameCounter % 5 != 0) {
      // Process every 5th frame only
      return;
    }
    // Forward the image to the cubit for processing
    if (mounted) {
      context.read<SegmentationCubit>().processCameraImage(cameraImage);
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return BlocConsumer<SegmentationCubit, SegmentationState>(
      // Only listen when the status changes.
      listenWhen: (previous, current) => previous.status != current.status,
      listener: (context, state) {
        if (state.status == SegmentationStatus.error &&
            state.errorMessage != null) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('❌ ${state.errorMessage}'),
              backgroundColor: Colors.red,
            ),
          );
        } else if (state.status == SegmentationStatus.modelReady) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('✅ Models Loaded & Ready!')),
          );
        }
      },
      builder: (context, state) {
        final cubit = context.read<SegmentationCubit>();
        final bool isModelReady =
            state.status == SegmentationStatus.modelReady &&
            state.errorMessage == null;
        return Scaffold(
          appBar: AppBar(
            title: const Text('Few-Shot Segmentation'),
            actions: [
              // IconButton(
              //   icon: const Icon(Icons.add_box_outlined),
              //   tooltip: 'Manage Models',
              //   onPressed: () async {
              //     // Wait for the result from ModelManagerScreen
              //     final newModelPath = await Navigator.of(context).push<String>(
              //       MaterialPageRoute(
              //         builder: (context) => const ModelManagerScreen(),
              //       ),
              //     );

              //     // If a path was returned, load it!
              //     if (newModelPath != null && context.mounted) {
              //       context.read<SegmentationCubit>().loadCustomClassifier(
              //         newModelPath,
              //       );

              //       ScaffoldMessenger.of(context).showSnackBar(
              //         const SnackBar(
              //           content: Text('🚀 New model loaded successfully!'),
              //         ),
              //       );
              //     }
              //   },
              // ),
              PopupMenuButton<int>(
                initialValue: state.selectedInputSize,
                tooltip: 'Select Input Size',
                onSelected: cubit.updateInputSize,
                itemBuilder: (BuildContext context) {
                  return [320, 400, 512, 768].map((int size) {
                    return PopupMenuItem<int>(
                      value: size,
                      child: Text('Input Size: $size'),
                    );
                  }).toList();
                },
                icon: const Icon(Icons.aspect_ratio),
              ),
              IconButton(
                icon: Icon(
                  Icons.tune,
                  color: state.showSlider
                      ? Theme.of(context).colorScheme.secondary
                      : null,
                ),
                tooltip: 'Toggle Threshold Slider',
                onPressed: cubit.toggleSliderVisibility,
              ),
            ],
          ),
          body: FutureBuilder<void>(
            future: _initializeControllerFuture,
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.done) {
                return Stack(
                  fit: StackFit.expand,
                  children: [
                    CameraPreview(_controller),
                    // Segmentation Overlay
                    if (state.overlayImage != null)
                      FittedBox(
                        fit: BoxFit.cover,
                        child: SizedBox(
                          width: state.overlayImage!.width.toDouble(),
                          height: state.overlayImage!.height.toDouble(),
                          child: CustomPaint(
                            painter: OverlayPainter(state.overlayImage!),
                          ),
                        ),
                      ),
                    // Loading Indicator (Initial Load)
                    if (state.status == SegmentationStatus.loadingModel)
                      Container(
                        color: Colors.black54,
                        child: const Center(
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              CircularProgressIndicator(),
                              SizedBox(height: 16),
                              Text(
                                "Loading Models...",
                                style: TextStyle(color: Colors.white),
                              ),
                            ],
                          ),
                        ),
                      ),
                    // Threshold Slider
                    if (state.showSlider)
                      Positioned(
                        bottom: 120,
                        left: 20,
                        right: 20,
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 16),
                          decoration: BoxDecoration(
                            color: Colors.black.withValues(alpha: 0.5),
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Row(
                            children: [
                              const Icon(Icons.tune, color: Colors.white),
                              Expanded(
                                child: Slider(
                                  value: state.similarityThreshold,
                                  min: 0.1, // Logistic regression can go lower
                                  max: 0.9,
                                  divisions: 8,
                                  label: state.similarityThreshold
                                      .toStringAsFixed(2),
                                  onChanged: cubit.updateThreshold,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                  ],
                );
              } else {
                return const Center(child: CircularProgressIndicator());
              }
            },
          ),
          floatingActionButton: Column(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              // Manage Models Button
              FloatingActionButton(
                heroTag: 'manage_models_btn', // Unique Tag 1
                onPressed: () async {
                  // Navigate to the Model Manager Screen
                  final newModelPath = await Navigator.of(context).push<String>(
                    MaterialPageRoute(
                      builder: (context) => const ModelManagerScreen(),
                    ),
                  );

                  // If the user downloaded a model, load it into the Cubit
                  if (newModelPath != null && context.mounted) {
                    context.read<SegmentationCubit>().loadCustomClassifier(
                      newModelPath,
                    );

                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(
                        content: Text('🚀 New model loaded successfully!'),
                      ),
                    );
                  }
                },
                tooltip: 'Manage Models',
                child: const Icon(Icons.library_add),
              ),
              const SizedBox(height: 16),

              // Play / Stop Button
              FloatingActionButton(
                heroTag: 'toggle_segmentation_btn',
                onPressed: isModelReady
                    ? cubit.toggleSegmentation
                    : () {
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            content: Text(
                              state.errorMessage ??
                                  '⚠️ Please load a model first!',
                            ),
                          ),
                        );
                      },
                backgroundColor: (state.errorMessage == null)
                    ? null
                    : Colors.grey,
                child: Icon(state.isSegmenting ? Icons.stop : Icons.play_arrow),
              ),
              const SizedBox(height: 16),

              // Filter Button
              FloatingActionButton(
                heroTag: 'filter_area_btn', // Unique Tag 3
                onPressed: cubit.toggleLargestAreaOnly,
                tooltip: 'Toggle Largest Area Only',
                backgroundColor: state.showLargestAreaOnly
                    ? Theme.of(context).primaryColor
                    : Colors.grey,
                child: const Icon(Icons.filter_center_focus),
              ),
            ],
          ),
        );
      },
    );
  }
}

class OverlayPainter extends CustomPainter {
  final ui.Image image;
  OverlayPainter(this.image);

  @override
  void paint(Canvas canvas, Size size) {
    paintImage(
      canvas: canvas,
      rect: Rect.fromLTWH(0, 0, size.width, size.height),
      image: image,
      fit: BoxFit.fill,
    );
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
