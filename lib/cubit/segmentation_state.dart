part of 'segmentation_cubit.dart';

enum SegmentationStatus { initial, loadingModel, modelReady, error }

@immutable
class SegmentationState extends Equatable {
  const SegmentationState({
    this.status = SegmentationStatus.initial,
    this.overlayImage,
    this.isSegmenting = false,
    this.similarityThreshold = 0.7,
    this.selectedInputSize = 320,
    this.showLargestAreaOnly = false,
    this.showSlider = false,
    this.errorMessage,
  });

  final SegmentationStatus status;
  final ui.Image? overlayImage;
  final bool isSegmenting;
  final double similarityThreshold;
  final int selectedInputSize;
  final bool showLargestAreaOnly;
  final bool showSlider;
  final String? errorMessage;

  SegmentationState copyWith({
    SegmentationStatus? status,
    ui.Image? overlayImage,
    bool? isSegmenting,
    double? similarityThreshold,
    int? selectedInputSize,
    bool? showLargestAreaOnly,
    bool? showSlider,
    String? errorMessage,
    bool clearOverlay = false,
  }) {
    return SegmentationState(
      status: status ?? this.status,
      overlayImage: clearOverlay ? null : overlayImage ?? this.overlayImage,
      isSegmenting: isSegmenting ?? this.isSegmenting,
      similarityThreshold: similarityThreshold ?? this.similarityThreshold,
      selectedInputSize: selectedInputSize ?? this.selectedInputSize,
      showLargestAreaOnly: showLargestAreaOnly ?? this.showLargestAreaOnly,
      showSlider: showSlider ?? this.showSlider,
      errorMessage: errorMessage ?? this.errorMessage,
    );
  }

  @override
  List<Object?> get props => [
    status,
    overlayImage,
    isSegmenting,
    similarityThreshold,
    selectedInputSize,
    showLargestAreaOnly,
    showSlider,
    errorMessage,
  ];
}
