// lib/services/pocketbase_service.dart

import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:pocketbase/pocketbase.dart';

// ⚠️ REPLACE with your actual PocketBase URL
const String kPocketBaseUrl = 'http://192.168.1.150:8090';

class PocketBaseService {
  // Singleton pattern to share the authenticated instance across screens
  static final PocketBaseService _instance = PocketBaseService._internal();

  late final PocketBase pb;

  factory PocketBaseService() {
    return _instance;
  }

  PocketBaseService._internal() {
    pb = PocketBase(kPocketBaseUrl);
  }

  /// Returns true if the user is currently authenticated
  bool get isAuthenticated => pb.authStore.isValid;

  /// Authenticates with email and password
  Future<void> login(String email, String password) async {
    await pb.collection('users').authWithPassword(email, password);
  }

  /// Uploads a new training dataset
  Future<RecordModel> createDataset(String name, List<XFile> images) async {
    // 1. Convert XFiles to MultipartFiles
    final List<http.MultipartFile> files = [];
    for (var image in images) {
      final bytes = await image.readAsBytes();
      final multipart = http.MultipartFile.fromBytes(
        'images',
        bytes,
        filename: image.name,
      );
      files.add(multipart);
    }

    // 2. Create Record
    final record = await pb
        .collection('datasets')
        .create(
          body: {
            'name': name,
            'status':
                'pending', // (Optional) Handled by default value in PB Schema
          },
          files: files,
        );

    return record;
  }

  /// Downloads the trained classifier.onnx from a record
  Future<File> downloadModelFile(RecordModel record) async {
    // Get the URL
    final filename = record.getStringValue('classifier_file');
    if (filename.isEmpty) throw Exception("No classifier file found in record");

    final url = pb.getFileUrl(record, filename);

    // Download bytes
    final response = await http.get(url);
    if (response.statusCode != 200) {
      throw Exception("Failed to download model: ${response.statusCode}");
    }

    // Save to App Documents
    final dir = await getApplicationDocumentsDirectory();
    // Use record ID in filename to avoid caching issues
    final savePath = '${dir.path}/classifier_${record.id}.onnx';
    final file = File(savePath);
    await file.writeAsBytes(response.bodyBytes);

    return file;
  }
}
