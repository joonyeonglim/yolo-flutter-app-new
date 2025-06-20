// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo_result.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:ultralytics_yolo/yolo_task.dart';
import 'package:ultralytics_yolo/yolo.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:share_plus/share_plus.dart';
import 'video_gallery_screen.dart';

/// 새 분류 및 녹화 화면
///
/// 이 화면은 다음 기능을 제공합니다:
/// - 실시간 카메라 피드와 YOLO 새 분류 (best-n-640-20250613 모델 고정 사용)
/// - 비디오 녹화 기능 (시작/중지)
/// - 조정 가능한 confidence 임계값
/// - 카메라 제어 (전환, 줌)  
/// - 성능 메트릭 (FPS)
/// - Top-N 분류 결과 표시
/// - 갤러리 연동으로 녹화된 비디오 관리
class CameraRecordingScreen extends StatefulWidget {
  const CameraRecordingScreen({super.key});

  @override
  State<CameraRecordingScreen> createState() => _CameraRecordingScreenState();
}

class _CameraRecordingScreenState extends State<CameraRecordingScreen>
    with WidgetsBindingObserver {
  
  // 로그 레벨 제어
  static const bool _enableDebugLogs = false; // 개발 시에만 true로 설정
  static const bool _enableInfoLogs = true;   // 중요한 정보만
  
  static void _debugLog(String message) {
    if (_enableDebugLogs) {
      debugPrint('[YOLO DEBUG] $message');
    }
  }
  
  static void _infoLog(String message) {
    if (_enableInfoLogs) {
      debugPrint('[YOLO INFO] $message');
    }
  }
  
  static void _errorLog(String message) {
    debugPrint('[YOLO ERROR] $message'); // 에러는 항상 출력
  }

  List<YOLOResult> _classificationResults = [];
  double _confidenceThreshold = 0.1;
  double _iouThreshold = 0.45;
  int _numItemsThreshold = 30;
  double _currentFps = 0.0;
  int _frameCount = 0;
  DateTime _lastFpsUpdate = DateTime.now();
  int _detectionCount = 0;

  bool _isModelLoading = false;
  double _currentZoomLevel = 1.0;
  String? _modelPath;
  String _loadingMessage = '';
  bool _isFrontCamera = false;

  // Recording 관련 변수들
  bool _isRecording = false;
  bool _isProcessingRecording = false;
  String? _recordingPath;
  bool _isCameraReady = false; // 카메라 준비 상태
  Duration _recordingDuration = Duration.zero;
  DateTime? _recordingStartTime;
  String? _currentRecordingPath;
  Timer? _recordingTimer;
  Timer? _maxDurationTimer;
  bool _isTogglingRecording = false;
  DateTime? _lastInactiveTime;
  bool _wasRecordingBeforeInactive = false;
  Timer? _lifecycleRecoveryTimer;

  final _yoloController = YOLOViewController();
  final _yoloViewKey = GlobalKey<YOLOViewState>();
  final bool _useController = true;

  // 고정된 모델 경로 - best-n-640-20250613 사용
  static const String _fixedModelPath = 'best-n-640-20250613';

  // 화면 방향 확인
  bool get isLandscape => MediaQuery.of(context).orientation == Orientation.landscape;

  /// 저장된 비디오를 갤러리에 등록
  Future<void> _saveVideoToGallery(String videoPath) async {
    try {
      if (Platform.isAndroid) {
        // Android: SharedPreferences에 비디오 경로 추가
        final prefs = await SharedPreferences.getInstance();
        final List<String> videoPaths = prefs.getStringList('recorded_videos') ?? [];
        
        // 중복 방지
        if (!videoPaths.contains(videoPath)) {
          videoPaths.add(videoPath);
          await prefs.setStringList('recorded_videos', videoPaths);
          _infoLog('✅ 비디오 경로가 갤러리에 추가됨: $videoPath');
        }
      }
      // iOS는 파일 시스템에서 자동으로 감지됨 (video_gallery_screen.dart에서 디렉토리 스캔)
    } catch (e) {
      _errorLog('❌ 갤러리 저장 실패: $e');
    }
  }

  /// 메시지 표시
  void _showMessage(String message) {
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(message),
          duration: const Duration(seconds: 2),
        ),
      );
    }
  }

  /// 카메라 워밍업
  Future<void> _warmUpCamera() async {
    try {
      _infoLog('🔥 카메라 워밍업 시작');
      
      // 카메라 초기화를 위한 작은 지연
      await Future.delayed(const Duration(milliseconds: 500));
      
      if (mounted && _useController) {
        // 초기 임계값 설정
        await _yoloController.setThresholds(
          confidenceThreshold: _confidenceThreshold,
          iouThreshold: _iouThreshold,
          numItemsThreshold: _numItemsThreshold,
        );
        
        _infoLog('✅ 카메라 워밍업 완료');
      }
    } catch (e) {
      _errorLog('❌ 카메라 워밍업 실패: $e');
    }
  }

  /// 모델 로드
  Future<void> _loadModel() async {
    setState(() {
      _isModelLoading = true;
      _loadingMessage = 'Loading bird classification model...';
    });

    try {
      // 모델 경로 설정
      setState(() {
        _modelPath = _fixedModelPath;
        _isModelLoading = false;
        _loadingMessage = '';
        _isCameraReady = true;
      });

      // Warm up camera after model is loaded and view is likely built
      WidgetsBinding.instance.addPostFrameCallback((_) {
        _warmUpCamera();
      });

    } catch (e) {
      _errorLog('Error loading model: $e');
      if (mounted) {
        setState(() {
          _isModelLoading = false;
          _loadingMessage = 'Failed to load model';
        });
        
        showDialog(
          context: context,
          builder: (context) => AlertDialog(
            title: const Text('Model Loading Error'),
            content: Text('Failed to load bird classification model: ${e.toString()}'),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('OK'),
              ),
            ],
          ),
        );
      }
    }
  }

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    
    // 초기 상태 명시적으로 설정
    _isRecording = false;
    _isProcessingRecording = false;
    _isCameraReady = false;
    
    _loadModel();

    // Set initial threshold after frame
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_useController) {
        _yoloController.setThresholds(
          confidenceThreshold: _confidenceThreshold,
          iouThreshold: 0.45, // IoU는 classification에서 사용되지 않지만 기본값 설정
          numItemsThreshold: 30,
        );
      } else {
        _yoloViewKey.currentState?.setThresholds(
          confidenceThreshold: _confidenceThreshold,
          iouThreshold: 0.45,
          numItemsThreshold: 30,
        );
      }
    });

    // 주기적으로 녹화 상태 동기화 (2초마다로 조정)
    Timer.periodic(const Duration(seconds: 2), (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }
      _syncRecordingState();
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _clearAllTimers();
    super.dispose();
  }

  /// 녹화 상태를 주기적으로 동기화
  Future<void> _syncRecordingState() async {
    // 처리 중이거나 카메라가 준비되지 않은 경우 스킵
    if (_isProcessingRecording || !_isCameraReady) return;
    
    try {
      final actualState = await _yoloController.isCurrentlyRecording();
      if (mounted && _isRecording != actualState) {
        _debugLog('🔄 상태 동기화: UI($_isRecording) → Native($actualState)');
        setState(() {
          _isRecording = actualState;
        });
        
        // 상태 변경 시 사용자에게 알림 (덜 방해가 되도록 조정)
        if (actualState) {
          _debugLog('📢 녹화 자동 시작 감지');
        } else {
          _debugLog('📢 녹화 자동 중지 감지');
        }
      }
    } catch (e) {
      // 상태 확인 실패 시 조용히 처리
      _debugLog('⚠️ 상태 동기화 실패: $e');
      
      // 연속 실패 시 강제 정리 시도
      if (_isRecording) {
        _debugLog('🔧 연속 실패로 인한 강제 상태 정리 시도');
        try {
          await _yoloController.stopRecording();
          setState(() {
            _isRecording = false;
            _recordingDuration = Duration.zero;
            _recordingStartTime = null;
            _currentRecordingPath = null;
          });
        } catch (forceError) {
          _debugLog('❌ 강제 정리도 실패: $forceError');
        }
      }
    }
  }

  /// Called when new classification results are available
  void _onClassificationResults(List<YOLOResult> results) {
    if (!mounted) return;

    _frameCount++;
    final now = DateTime.now();
    final elapsed = now.difference(_lastFpsUpdate).inMilliseconds;

    if (elapsed >= 1000) {
      final calculatedFps = _frameCount * 1000 / elapsed;
      _currentFps = calculatedFps;
      _frameCount = 0;
      _lastFpsUpdate = now;
    }

    // Filter and sort results
    final filteredResults = results
        .where((r) => r.confidence >= _confidenceThreshold)
        .toList();
    
    // Sort by confidence (highest first)
    filteredResults.sort((a, b) => b.confidence.compareTo(a.confidence));

    setState(() {
      _classificationResults = filteredResults.take(5).toList(); // Top 5 results
      _detectionCount = filteredResults.length;
    });
  }

  /// Recording 시작/중지 (네이티브 상태 기반)
  Future<void> _toggleRecording() async {
    // 중복 요청 및 카메라 미준비 상태 방지
    if (_isProcessingRecording || !_isCameraReady) {
      _debugLog('녹화 토글 무시: 처리 중($_isProcessingRecording) 또는 카메라 미준비(!$_isCameraReady)');
      return;
    }
    
    setState(() {
      _isProcessingRecording = true;
    });

    try {
      final isCurrentlyRecording = await _yoloController.isCurrentlyRecording();
      _debugLog('네이티브 녹화 상태: $isCurrentlyRecording, UI 상태: $_isRecording');

      if (isCurrentlyRecording) {
        // ========== 녹화 중지 ==========
        _infoLog('🛑 녹화 중지 시도');
        
        // 타이머 정리
        _clearRecordingTimers();
        
        final result = await _yoloController.stopRecording();
        
        if (result != null && result.isNotEmpty) {
          _infoLog('✅ 녹화 중지 성공: $result');
          
          // 갤러리에 비디오 저장
          await _saveVideoToGallery(result);
          
          _showMessage('녹화가 저장되었습니다.');
        } else {
          _errorLog('⚠️ 녹화 중지 성공했으나 결과 없음');
          _showMessage('녹화가 중지되었지만 저장에 실패했습니다.');
        }
        
        // 상태 동기화
        setState(() {
          _isRecording = false;
          _recordingDuration = Duration.zero;
          _recordingStartTime = null;
          _currentRecordingPath = null;
        });

      } else {
        // ========== 녹화 시작 ==========
        _infoLog('▶️ 녹화 시작 시도');
        await _yoloController.startRecording();
        _infoLog('✅ 녹화 시작 성공');
        _showMessage('녹화가 시작되었습니다.');

        // 녹화 시간 타이머 시작
        _recordingStartTime = DateTime.now();
        _startRecordingTimer();

        // 상태 동기화
        setState(() {
          _isRecording = true;
          _recordingDuration = Duration.zero;
        });
      }
    } catch (e) {
      _errorLog('❌ 녹화 토글 중 예외 발생: $e');
      _showMessage('오류가 발생했습니다: ${e.toString()}');
      
      // 오류 발생 시 상태 재동기화
      await _syncRecordingState();

    } finally {
      setState(() {
        _isProcessingRecording = false;
      });
      _debugLog('=== 녹화 토글 완료 ===');
    }
  }

  /// 녹화된 비디오 공유
  void _shareRecording(String path) {
    try {
      Share.shareXFiles([XFile(path)], text: 'YOLO 새 분류 영상');
    } catch (e) {
      _errorLog('Sharing video failed: $e');
    }
  }

  /// 녹화 시간 타이머 시작
  void _startRecordingTimer() {
    _clearRecordingTimers(); // 기존 타이머 정리
    
    _recordingTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (!mounted || !_isRecording || _recordingStartTime == null) {
        timer.cancel();
        return;
      }
      
      final now = DateTime.now();
      final duration = now.difference(_recordingStartTime!);
      
      setState(() {
        _recordingDuration = duration;
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // YOLO View
          if (_modelPath != null && !_isModelLoading)
            YOLOView(
              key: _useController
                  ? const ValueKey('yolo_view_static')
                  : _yoloViewKey,
              controller: _useController ? _yoloController : null,
              modelPath: _modelPath!,
              task: YOLOTask.classify, // Classification task
              onResult: _onClassificationResults,
              onPerformanceMetrics: (metrics) {
                if (mounted) {
                  setState(() {
                    _currentFps = metrics.fps;
                  });
                }
              },
              onZoomChanged: (zoomLevel) {
                if (mounted) {
                  setState(() {
                    _currentZoomLevel = zoomLevel;
                  });
                }
              },
            )
          else if (_isModelLoading)
            _buildLoadingScreen(),

          // 상단 정보 (감지 수, FPS, 녹화 상태)
          Positioned(
            top: MediaQuery.of(context).padding.top + (isLandscape ? 8 : 16),
            left: isLandscape ? 8 : 16,
            right: isLandscape ? 8 : 16,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                // FPS 정보만 표시  
                IgnorePointer(
                  child: Text(
                    'FPS: ${_currentFps.toStringAsFixed(1)}',
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w600,
                      fontSize: 16,
                    ),
                  ),
                ),
                
                // 녹화 시간 표시
                const SizedBox(height: 8),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 6,
                  ),
                  decoration: BoxDecoration(
                    color: _isRecording 
                      ? Colors.red.withOpacity(0.7) 
                      : Colors.black54.withOpacity(0.7),
                    borderRadius: BorderRadius.circular(50),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      if (_isRecording) ...[
                        Container(
                          width: 8,
                          height: 8,
                          decoration: const BoxDecoration(
                            color: Colors.white,
                            shape: BoxShape.circle,
                          ),
                        ),
                        const SizedBox(width: 8),
                      ],
                      Text(
                        _isRecording
                          ? '${(_recordingDuration.inMinutes).toString().padLeft(2, '0')}:${(_recordingDuration.inSeconds % 60).toString().padLeft(2, '0')}'
                          : '00:00',
                        style: const TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                        ),
                      ),
                    ],
                  ),
                ),
                
                const SizedBox(height: 8),
              ],
            ),
          ),

          // 녹화 버튼
          Positioned(
            bottom: isLandscape ? 24 : 40,
            left: 0,
            right: 0,
            child: Center(
              child: GestureDetector(
                onTap: _toggleRecording,
                child: Container(
                  width: 80,
                  height: 80,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: _isRecording ? Colors.red : Colors.white,
                    border: Border.all(
                      color: _isRecording ? Colors.white : Colors.red,
                      width: 4,
                    ),
                  ),
                  child: Center(
                    child: Container(
                      width: _isRecording ? 24 : 32,
                      height: _isRecording ? 24 : 32,
                      decoration: BoxDecoration(
                        color: _isRecording ? Colors.white : Colors.red,
                        shape: _isRecording ? BoxShape.rectangle : BoxShape.circle,
                        borderRadius: _isRecording ? BorderRadius.circular(4) : null,
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),



          // 카메라 전환 버튼 (우상단)
          Positioned(
            top: MediaQuery.of(context).padding.top + (isLandscape ? 8 : 16),
            right: isLandscape ? 8 : 16,
            child: CircleAvatar(
              radius: isLandscape ? 20 : 24,
              backgroundColor: Colors.black.withValues(alpha: 0.5),
              child: IconButton(
                icon: const Icon(Icons.flip_camera_ios, color: Colors.white),
                onPressed: () {
                  setState(() {
                    _isFrontCamera = !_isFrontCamera;
                  });
                  if (_useController) {
                    _yoloController.switchCamera();
                  } else {
                    _yoloViewKey.currentState?.switchCamera();
                  }
                },
              ),
            ),
          ),

          // 프레임 레이트 관리 버튼 (좌상단)
          Positioned(
            top: MediaQuery.of(context).padding.top + (isLandscape ? 8 : 16) + 60,
            left: isLandscape ? 8 : 16,
            child: CircleAvatar(
              radius: isLandscape ? 18 : 20,
              backgroundColor: Colors.black.withValues(alpha: 0.5),
              child: IconButton(
                icon: const Icon(Icons.speed, color: Colors.white, size: 20),
                onPressed: _showFrameRateSettings,
              ),
            ),
          ),

          // Bottom left buttons (갤러리 및 카메라 전환)
          Positioned(
            bottom: 32,
            left: 16,
            child: Column(
              children: [
                // Gallery button
                CircleAvatar(
                  radius: 24,
                  backgroundColor: Colors.black.withValues(alpha: 0.5),
                  child: IconButton(
                    icon: const Icon(Icons.video_library, color: Colors.white),
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => const VideoGalleryScreen(),
                        ),
                      );
                    },
                  ),
                ),
                const SizedBox(height: 12),
                // Camera flip button
                CircleAvatar(
                  radius: 24,
                  backgroundColor: Colors.black.withValues(alpha: 0.5),
                  child: IconButton(
                    icon: const Icon(Icons.flip_camera_ios, color: Colors.white),
                    onPressed: () {
                      setState(() {
                        _isFrontCamera = !_isFrontCamera;
                      });
                      if (_useController) {
                        _yoloController.switchCamera();
                      } else {
                        _yoloViewKey.currentState?.switchCamera();
                      }
                    },
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  /// 로딩 화면을 구성합니다
  Widget _buildLoadingScreen() {
    return IgnorePointer(
      child: Container(
        color: Colors.black87,
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Ultralytics 로고
              Image.asset(
                'assets/logo.png',
                width: 120,
                height: 120,
                color: Colors.white.withValues(alpha: 0.8),
              ),
              const SizedBox(height: 32),
              // 로딩 메시지
              Text(
                _loadingMessage,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.w500,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 24),
              // 진행률 표시기
              const CircularProgressIndicator(color: Colors.white),
            ],
          ),
        ),
      ),
    );
  }







  /// 오류 대화상자를 표시합니다
  void _showErrorDialog(String title, String content) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(title),
        content: Text(content),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  /// 프레임 레이트 설정 다이얼로그를 표시합니다
  void _showFrameRateSettings() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Frame Rate Settings'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text('Select recording frame rate:'),
              const SizedBox(height: 16),
              ...['30 FPS', '60 FPS', '120 FPS'].map((fps) {
                return ListTile(
                  title: Text(fps),
                  onTap: () async {
                    Navigator.pop(context);
                    final frameRate = int.parse(fps.split(' ')[0]);
                    await _setFrameRate(frameRate);
                  },
                );
              }),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel'),
            ),
          ],
        );
      },
    );
  }

  /// 프레임 레이트를 설정합니다
  Future<void> _setFrameRate(int frameRate) async {
    try {
      if (_useController) {
        await _yoloController.setFrameRate(frameRate);
        debugPrint('Frame rate set to: $frameRate FPS');
        
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Frame rate set to $frameRate FPS'),
              duration: const Duration(seconds: 2),
            ),
          );
        }
      }
    } catch (e) {
      debugPrint('Failed to set frame rate: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to set frame rate: $e'),
            backgroundColor: Colors.red,
            duration: const Duration(seconds: 2),
          ),
        );
      }
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    super.didChangeAppLifecycleState(state);
    debugPrint('App lifecycle state changed: $state');

    if (!mounted) return;

    switch (state) {
      case AppLifecycleState.inactive:
      case AppLifecycleState.paused:
        debugPrint('App paused/inactive: handling recording state');
        _lastInactiveTime = DateTime.now();
        _wasRecordingBeforeInactive = _isRecording;

        if (_isRecording) {
          debugPrint('Recording during interruption: stopping recording');
          _stopRecording();
        }
        break;

      case AppLifecycleState.resumed:
        debugPrint('App resumed: checking recovery needs');
        _lifecycleRecoveryTimer?.cancel();
        
        Duration? interruptionDuration;
        if (_lastInactiveTime != null) {
          interruptionDuration = DateTime.now().difference(_lastInactiveTime!);
          debugPrint('Interruption duration: ${interruptionDuration.inSeconds} seconds');
        }

        Future.delayed(const Duration(milliseconds: 100), () {
          if (mounted) {
            _handleAppResume(interruptionDuration);
          }
        });
        break;

      case AppLifecycleState.detached:
        debugPrint('App detached: cleaning up resources');
        _clearAllTimers();
        break;

      default:
        break;
    }
  }

  /// 앱 재개 시 복구 처리를 수행합니다
  Future<void> _handleAppResume(Duration? interruptionDuration) async {
    try {
      debugPrint('Handling app resume recovery');

      // 상태 초기화
      _lastInactiveTime = null;
      _wasRecordingBeforeInactive = false;

      debugPrint('App resume recovery completed');
    } catch (e) {
      debugPrint('Error during app resume recovery: $e');
    }
  }

  /// 녹화를 중지합니다
  Future<void> _stopRecording() async {
    if (!_isRecording) {
      debugPrint('Stop recording: not currently recording');
      _clearRecordingTimers();
      return;
    }

    debugPrint('Stopping recording...');
    try {
      // 타이머 즉시 정리
      _clearRecordingTimers();

      // YOLO 컨트롤러를 통한 녹화 중지
      String? videoPath;
      if (_useController) {
        videoPath = await _yoloController.stopRecording();
        debugPrint('Recording stopped, video saved to: $videoPath');
      }

      if (mounted) {
        setState(() {
          _isRecording = false;
          _recordingDuration = Duration.zero;
          _recordingStartTime = null;
          _currentRecordingPath = null;
        });
      }

      // 비디오가 저장되었을 때 사용자에게 알림
      if (videoPath != null && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Video saved to: ${videoPath.split('/').last}'),
            duration: const Duration(seconds: 2),
          ),
        );
      }

      debugPrint('Recording stopped successfully');
    } catch (e) {
      debugPrint('Failed to stop recording: $e');
      _clearRecordingTimers();
      
      if (mounted) {
        setState(() {
          _isRecording = false;
          _recordingDuration = Duration.zero;
          _recordingStartTime = null;
          _currentRecordingPath = null;
        });
      }
    }
  }

  /// 녹화 관련 타이머들을 정리합니다
  void _clearRecordingTimers() {
    _recordingTimer?.cancel();
    _recordingTimer = null;
    _maxDurationTimer?.cancel();
    _maxDurationTimer = null;
  }

  /// 모든 타이머들을 정리합니다
  void _clearAllTimers() {
    _clearRecordingTimers();
    _lifecycleRecoveryTimer?.cancel();
    _lifecycleRecoveryTimer = null;
  }

  /// 녹화 에러에서 복구합니다
  void _recoverFromRecordingError() {
    try {
      if (_isRecording) {
        _clearRecordingTimers();
        setState(() {
          _isRecording = false;
          _recordingDuration = Duration.zero;
          _recordingStartTime = null;
          _currentRecordingPath = null;
        });
      }
    } catch (e) {
      debugPrint('Error during recording recovery: $e');
    }
  }
} 