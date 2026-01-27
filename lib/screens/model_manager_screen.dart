// lib/screens/model_manager_screen.dart

import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:pocketbase/pocketbase.dart';
import '../services/pocketbase_service.dart';

class ModelManagerScreen extends StatefulWidget {
  const ModelManagerScreen({super.key});

  @override
  State<ModelManagerScreen> createState() => _ModelManagerScreenState();
}

class _ModelManagerScreenState extends State<ModelManagerScreen> {
  final _nameController = TextEditingController();
  final _picker = ImagePicker();
  final _pbService = PocketBaseService();

  List<XFile> _selectedImages = [];
  bool _isUploading = false;
  Timer? _pollTimer;
  RecordModel? _currentRecord;
  String _status = 'idle'; // idle, pending, training, ready, failed

  @override
  void initState() {
    super.initState();
    // Check auth immediately upon entering the screen
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _checkAuth();
    });
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    super.dispose();
  }

  Future<void> _checkAuth() async {
    if (!_pbService.isAuthenticated) {
      await _showLoginDialog();
    }
  }

  Future<void> _showLoginDialog() async {
    final emailController = TextEditingController();
    final passwordController = TextEditingController();

    await showDialog(
      context: context,
      barrierDismissible: false, // User must log in or go back
      builder: (context) => AlertDialog(
        title: const Text('Login Required'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: emailController,
              decoration: const InputDecoration(labelText: 'Email'),
              keyboardType: TextInputType.emailAddress,
            ),
            TextField(
              controller: passwordController,
              decoration: const InputDecoration(labelText: 'Password'),
              obscureText: true,
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.pop(context); // Close dialog
              Navigator.pop(context); // Go back to Camera Screen
            },
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () async {
              try {
                await _pbService.login(
                  emailController.text,
                  passwordController.text,
                );
                if (mounted) Navigator.pop(context); // Close dialog on success
              } catch (e) {
                ScaffoldMessenger.of(
                  context,
                ).showSnackBar(SnackBar(content: Text('Login Failed: $e')));
              }
            },
            child: const Text('Login'),
          ),
        ],
      ),
    );
  }

  Future<void> _pickImages() async {
    final List<XFile> images = await _picker.pickMultiImage();
    if (images.isNotEmpty) {
      setState(() {
        _selectedImages.addAll(images);
      });
    }
  }

  void _removeImage(int index) {
    setState(() {
      _selectedImages.removeAt(index);
    });
  }

  Future<void> _uploadDataset() async {
    if (_nameController.text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter a dataset name')),
      );
      return;
    }

    if (_selectedImages.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please select at least one image')),
      );
      return;
    }

    setState(() => _isUploading = true);

    try {
      // Capture the record returned from the service
      final record = await _pbService.createDataset(
        _nameController.text,
        _selectedImages,
      );

      if (mounted) {
        // Start polling using the new record's ID
        _startPolling(record.id);

        // Update the local state so the UI switches to the status view
        setState(() {
          _currentRecord = record;
          _status = 'pending'; // This triggers the "Queued for training" view
          _selectedImages.clear();
          _nameController.clear();
        });

        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('✅ Upload Complete! Training pending...'),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('❌ Upload Failed: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _isUploading = false);
    }
  }

  void _startPolling(String recordId) {
    _pollTimer?.cancel();
    _pollTimer = Timer.periodic(const Duration(seconds: 3), (timer) async {
      try {
        final record = await _pbService.pb
            .collection('datasets')
            .getOne(recordId);
        final newStatus = record.getStringValue('status');

        if (mounted) {
          setState(() {
            _currentRecord = record;
            _status = newStatus;
          });
        }

        if (newStatus == 'ready' || newStatus == 'failed') {
          timer.cancel(); // Stop polling when done
        }
      } catch (e) {
        print("Polling error: $e");
      }
    });
  }

  Future<void> _downloadAndUse() async {
    if (_currentRecord == null) return;

    try {
      setState(() => _isUploading = true); // Reuse loading state

      // Download
      final file = await _pbService.downloadModelFile(_currentRecord!);

      if (mounted) {
        // Return path to CameraScreen
        Navigator.pop(context, file.path);
      }
    } catch (e) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Download failed: $e')));
    } finally {
      if (mounted) setState(() => _isUploading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_status != 'idle') {
      return Scaffold(
        appBar: AppBar(title: const Text('Training Status')),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (_status == 'pending') ...[
                const CircularProgressIndicator(),
                const SizedBox(height: 20),
                const Text('Queued for training...'),
              ] else if (_status == 'training') ...[
                const CircularProgressIndicator(),
                const SizedBox(height: 20),
                const Text('Training in progress...'),
                const Text(
                  '(This usually takes 10-30 seconds)',
                  style: TextStyle(color: Colors.grey),
                ),
              ] else if (_status == 'ready') ...[
                const Icon(Icons.check_circle, color: Colors.green, size: 80),
                const SizedBox(height: 20),
                Text(
                  'Model "${_nameController.text}" is Ready!',
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 30),
                FilledButton.icon(
                  onPressed: _downloadAndUse,
                  icon: const Icon(Icons.download),
                  label: const Text('Download & Use Model'),
                  style: FilledButton.styleFrom(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 32,
                      vertical: 16,
                    ),
                  ),
                ),
              ] else if (_status == 'failed') ...[
                const Icon(Icons.error, color: Colors.red, size: 80),
                const SizedBox(height: 20),
                const Text('Training Failed.'),
                const SizedBox(height: 20),
                OutlinedButton(
                  onPressed: () => setState(() => _status = 'idle'),
                  child: const Text('Try Again'),
                ),
              ],
            ],
          ),
        ),
      );
    }
    return Scaffold(
      appBar: AppBar(title: const Text('New Training Set')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // 1. Name Input
            TextField(
              controller: _nameController,
              decoration: const InputDecoration(
                labelText: 'Object Name (e.g., "Red Mug")',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.label),
              ),
            ),
            const SizedBox(height: 16),

            // 2. Image Grid
            Expanded(
              child: _selectedImages.isEmpty
                  ? Center(
                      child: Text(
                        'No images selected.\nTap + to add training images.',
                        textAlign: TextAlign.center,
                        style: Theme.of(
                          context,
                        ).textTheme.bodyLarge?.copyWith(color: Colors.grey),
                      ),
                    )
                  : GridView.builder(
                      gridDelegate:
                          const SliverGridDelegateWithFixedCrossAxisCount(
                            crossAxisCount: 3,
                            crossAxisSpacing: 8,
                            mainAxisSpacing: 8,
                          ),
                      itemCount: _selectedImages.length,
                      itemBuilder: (context, index) {
                        return Stack(
                          fit: StackFit.expand,
                          children: [
                            Image.file(
                              File(_selectedImages[index].path),
                              fit: BoxFit.cover,
                            ),
                            Positioned(
                              top: 2,
                              right: 2,
                              child: GestureDetector(
                                onTap: () => _removeImage(index),
                                child: Container(
                                  color: Colors.black54,
                                  child: const Icon(
                                    Icons.close,
                                    color: Colors.white,
                                    size: 20,
                                  ),
                                ),
                              ),
                            ),
                          ],
                        );
                      },
                    ),
            ),

            // Action Buttons
            const SizedBox(height: 16),
            if (_isUploading)
              const Center(child: CircularProgressIndicator())
            else
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: _pickImages,
                      icon: const Icon(Icons.add_photo_alternate),
                      label: const Text('Add Images'),
                      style: OutlinedButton.styleFrom(
                        padding: const EdgeInsets.all(16),
                      ),
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: FilledButton.icon(
                      onPressed: _uploadDataset,
                      icon: const Icon(Icons.cloud_upload),
                      label: const Text('Upload & Train'),
                      style: FilledButton.styleFrom(
                        padding: const EdgeInsets.all(16),
                      ),
                    ),
                  ),
                ],
              ),
          ],
        ),
      ),
    );
  }
}
