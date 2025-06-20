import 'package:flutter/material.dart';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:video_player/video_player.dart';

class VideoGalleryScreen extends StatefulWidget {
  const VideoGalleryScreen({super.key});

  @override
  State<VideoGalleryScreen> createState() => _VideoGalleryScreenState();
}

class _VideoGalleryScreenState extends State<VideoGalleryScreen> {
  List<VideoFile> _videos = [];
  bool _isLoading = true;
  
  @override
  void initState() {
    super.initState();
    _loadVideos();
  }

  Future<void> _loadVideos() async {
    setState(() => _isLoading = true);
    
    try {
      final videos = await _getRecordedVideos();
      setState(() {
        _videos = videos;
        _isLoading = false;
      });
    } catch (e) {
      debugPrint('Error loading videos: $e');
      setState(() => _isLoading = false);
    }
  }

  Future<List<VideoFile>> _getRecordedVideos() async {
    final List<VideoFile> videos = [];
    
    if (Platform.isIOS) {
      // iOS: Documents 디렉토리와 임시 디렉토리에서 .mp4 파일 찾기
      try {
        // Documents 디렉토리 확인
        final documentsDir = await getApplicationDocumentsDirectory();
        await _scanDirectory(documentsDir, videos);
        
        // 임시 디렉토리도 확인 (기존 녹화 파일들)
        try {
          final tempDir = await getTemporaryDirectory();
          await _scanDirectory(tempDir, videos);
        } catch (e) {
          debugPrint('Error reading temp directory: $e');
        }
      } catch (e) {
        debugPrint('Error reading iOS videos: $e');
      }
    } else if (Platform.isAndroid) {
      // Android: SharedPreferences에서 저장된 비디오 URI들 가져오기
      try {
        final prefs = await SharedPreferences.getInstance();
        final List<String> videoPaths = prefs.getStringList('recorded_videos') ?? [];
        
        for (final path in videoPaths) {
          // 기본 정보만 표시 (Android에서는 URI로 파일 크기 등을 쉽게 가져올 수 없음)
          final fileName = path.split('/').last.replaceAll(':', '_');
          videos.add(VideoFile(
            path: path,
            name: fileName.isNotEmpty ? fileName : 'Video ${videos.length + 1}',
            size: 0, // Android URI는 크기 정보 없음
            createdAt: DateTime.now(), // 실제 생성일 대신 현재 시간 사용
          ));
        }
      } catch (e) {
        debugPrint('Error reading Android videos: $e');
      }
    }
    
    // 최신순으로 정렬 (iOS만 실제 날짜 정렬 가능)
    if (Platform.isIOS) {
      videos.sort((a, b) => b.createdAt.compareTo(a.createdAt));
    }
    
    return videos;
  }

  Future<void> _scanDirectory(Directory directory, List<VideoFile> videos) async {
    try {
      final List<FileSystemEntity> entities = directory.listSync();
      
      for (final entity in entities) {
        if (entity is File && entity.path.endsWith('.mp4')) {
          final stat = await entity.stat();
          final fileName = entity.path.split('/').last;
          
          // 중복 파일 확인 (같은 이름의 파일이 이미 있는지)
          if (!videos.any((v) => v.name == fileName)) {
            videos.add(VideoFile(
              path: entity.path,
              name: fileName,
              size: stat.size,
              createdAt: stat.modified,
            ));
          }
        }
      }
    } catch (e) {
      debugPrint('Error scanning directory ${directory.path}: $e');
    }
  }

  String _formatFileSize(int bytes) {
    if (bytes < 1024) return '$bytes B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(1)} KB';
    return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
  }

  String _formatDate(DateTime date) {
    return '${date.year}-${date.month.toString().padLeft(2, '0')}-${date.day.toString().padLeft(2, '0')} '
           '${date.hour.toString().padLeft(2, '0')}:${date.minute.toString().padLeft(2, '0')}';
  }

  Future<void> _playVideo(VideoFile video) async {
    try {
      // Flutter video_player로 비디오 재생
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => VideoPlayerScreen(videoPath: video.path),
        ),
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('비디오 재생 실패: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _deleteVideo(VideoFile video) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('비디오 삭제'),
        content: Text('${video.name}을(를) 삭제하시겠습니까?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('취소'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('삭제', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );

    if (confirmed == true) {
      try {
        if (Platform.isIOS) {
          // iOS: 실제 파일 삭제
          final file = File(video.path);
          if (await file.exists()) {
            await file.delete();
          }
        } else if (Platform.isAndroid) {
          // Android: SharedPreferences에서 경로 제거 (실제 파일은 시스템이 관리)
          final prefs = await SharedPreferences.getInstance();
          final List<String> videoPaths = prefs.getStringList('recorded_videos') ?? [];
          videoPaths.remove(video.path);
          await prefs.setStringList('recorded_videos', videoPaths);
        }
        
        await _loadVideos(); // 목록 새로고침
        
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('비디오가 삭제되었습니다')),
          );
        }
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('삭제 실패: $e'),
              backgroundColor: Colors.red,
            ),
          );
        }
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Video Gallery'),
            if (!_isLoading && _videos.isNotEmpty)
              Text(
                '${_videos.length}개의 비디오',
                style: const TextStyle(
                  fontSize: 12,
                  color: Colors.white70,
                  fontWeight: FontWeight.normal,
                ),
              ),
          ],
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadVideos,
            tooltip: '새로고침',
          ),
        ],
      ),
      body: _isLoading
          ? const Center(
              child: CircularProgressIndicator(color: Colors.white),
            )
          : _videos.isEmpty
              ? Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Container(
                        padding: const EdgeInsets.all(24),
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: Colors.grey[900],
                        ),
                        child: const Icon(
                          Icons.videocam_off,
                          size: 48,
                          color: Colors.white54,
                        ),
                      ),
                      const SizedBox(height: 24),
                      const Text(
                        '저장된 비디오가 없습니다',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 20,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(height: 8),
                      const Text(
                        '카메라에서 새 비디오를 녹화해보세요!',
                        style: TextStyle(
                          color: Colors.white60,
                          fontSize: 14,
                        ),
                      ),
                      const SizedBox(height: 24),
                      ElevatedButton.icon(
                        onPressed: () => Navigator.pop(context),
                        icon: const Icon(Icons.videocam, color: Colors.white),
                        label: const Text(
                          '카메라로 돌아가기',
                          style: TextStyle(color: Colors.white),
                        ),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.red.shade600,
                          padding: const EdgeInsets.symmetric(
                            horizontal: 24,
                            vertical: 12,
                          ),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(24),
                          ),
                        ),
                      ),
                    ],
                  ),
                )
              : RefreshIndicator(
                  onRefresh: _loadVideos,
                  child: ListView.builder(
                    padding: const EdgeInsets.all(16),
                    itemCount: _videos.length,
                    itemBuilder: (context, index) {
                      final video = _videos[index];
                      return Card(
                        color: Colors.grey[900],
                        margin: const EdgeInsets.only(bottom: 12),
                        elevation: 4,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: ListTile(
                          contentPadding: const EdgeInsets.all(12),
                          leading: Container(
                            width: 60,
                            height: 60,
                            decoration: BoxDecoration(
                              gradient: LinearGradient(
                                colors: [Colors.red.shade600, Colors.orange.shade600],
                                begin: Alignment.topLeft,
                                end: Alignment.bottomRight,
                              ),
                              borderRadius: BorderRadius.circular(10),
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.red.withOpacity(0.3),
                                  blurRadius: 8,
                                  offset: const Offset(0, 2),
                                ),
                              ],
                            ),
                            child: const Icon(
                              Icons.play_circle_fill,
                              color: Colors.white,
                              size: 28,
                            ),
                          ),
                          title: Text(
                            video.name.replaceAll('YOLO_Bird_', '').replaceAll('.mp4', ''),
                            style: const TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.w600,
                              fontSize: 15,
                            ),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                          subtitle: Padding(
                            padding: const EdgeInsets.only(top: 8),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  children: [
                                    Icon(
                                      Icons.access_time,
                                      size: 14,
                                      color: Colors.white60,
                                    ),
                                    const SizedBox(width: 4),
                                    Text(
                                      _formatDate(video.createdAt),
                                      style: const TextStyle(
                                        color: Colors.white70,
                                        fontSize: 12,
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 4),
                                Row(
                                  children: [
                                    Icon(
                                      Icons.storage,
                                      size: 14,
                                      color: Colors.white60,
                                    ),
                                    const SizedBox(width: 4),
                                    Text(
                                      _formatFileSize(video.size),
                                      style: const TextStyle(
                                        color: Colors.white54,
                                        fontSize: 12,
                                      ),
                                    ),
                                  ],
                                ),
                              ],
                            ),
                          ),
                          trailing: PopupMenuButton<String>(
                            iconColor: Colors.white,
                            color: Colors.grey[800],
                            onSelected: (value) {
                              switch (value) {
                                case 'play':
                                  _playVideo(video);
                                  break;
                                case 'delete':
                                  _deleteVideo(video);
                                  break;
                              }
                            },
                            itemBuilder: (context) => [
                              const PopupMenuItem(
                                value: 'play',
                                child: Row(
                                  children: [
                                    Icon(Icons.play_arrow, color: Colors.white),
                                    SizedBox(width: 8),
                                    Text('재생', style: TextStyle(color: Colors.white)),
                                  ],
                                ),
                              ),
                              const PopupMenuItem(
                                value: 'delete',
                                child: Row(
                                  children: [
                                    Icon(Icons.delete, color: Colors.red),
                                    SizedBox(width: 8),
                                    Text('삭제', style: TextStyle(color: Colors.red)),
                                  ],
                                ),
                              ),
                            ],
                          ),
                          onTap: () => _playVideo(video),
                        ),
                      );
                    },
                  ),
                ),
    );
  }
}

class VideoFile {
  final String path;
  final String name;
  final int size;
  final DateTime createdAt;

  VideoFile({
    required this.path,
    required this.name,
    required this.size,
    required this.createdAt,
  });
}

class VideoPlayerScreen extends StatefulWidget {
  final String videoPath;

  const VideoPlayerScreen({super.key, required this.videoPath});

  @override
  State<VideoPlayerScreen> createState() => _VideoPlayerScreenState();
}

class _VideoPlayerScreenState extends State<VideoPlayerScreen> {
  late VideoPlayerController _controller;
  bool _isLoading = true;
  bool _hasError = false;
  String _errorMessage = '';

  @override
  void initState() {
    super.initState();
    _initializePlayer();
  }

  Future<void> _initializePlayer() async {
    try {
      if (Platform.isIOS) {
        // iOS: 파일 경로로 직접 재생
        _controller = VideoPlayerController.file(File(widget.videoPath));
      } else {
        // Android: URI로 재생
        _controller = VideoPlayerController.networkUrl(Uri.parse(widget.videoPath));
      }

      await _controller.initialize();
      
      setState(() {
        _isLoading = false;
      });

      _controller.play();
    } catch (e) {
      setState(() {
        _isLoading = false;
        _hasError = true;
        _errorMessage = e.toString();
      });
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: const Text('Video Player'),
      ),
      body: Center(
        child: _isLoading
            ? const CircularProgressIndicator(color: Colors.white)
            : _hasError
                ? Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(
                        Icons.error,
                        color: Colors.red,
                        size: 64,
                      ),
                      const SizedBox(height: 16),
                      const Text(
                        '비디오를 재생할 수 없습니다',
                        style: TextStyle(color: Colors.white, fontSize: 18),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        _errorMessage,
                        style: const TextStyle(color: Colors.white54, fontSize: 14),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  )
                : AspectRatio(
                    aspectRatio: _controller.value.aspectRatio,
                    child: Stack(
                      children: [
                        VideoPlayer(_controller),
                        Positioned(
                          bottom: 0,
                          left: 0,
                          right: 0,
                          child: VideoProgressIndicator(
                            _controller,
                            allowScrubbing: true,
                            colors: const VideoProgressColors(
                              playedColor: Colors.red,
                              backgroundColor: Colors.grey,
                              bufferedColor: Colors.white30,
                            ),
                          ),
                        ),
                        Center(
                          child: IconButton(
                            iconSize: 64,
                            icon: Icon(
                              _controller.value.isPlaying
                                  ? Icons.pause_circle_filled
                                  : Icons.play_circle_filled,
                              color: Colors.white.withValues(alpha: 0.8),
                            ),
                            onPressed: () {
                              setState(() {
                                if (_controller.value.isPlaying) {
                                  _controller.pause();
                                } else {
                                  _controller.play();
                                }
                              });
                            },
                          ),
                        ),
                      ],
                    ),
                  ),
      ),
    );
  }
} 