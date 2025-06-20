// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//
//  This file is part of the Ultralytics YOLO Package, managing camera capture for real-time inference.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The VideoCapture component manages the camera and video processing pipeline for real-time
//  object detection. It handles setting up the AVCaptureSession, managing camera devices,
//  configuring camera properties like focus and exposure, and processing video frames for
//  model inference. The class delivers capture frames to the predictor component for real-time
//  analysis and returns results through delegate callbacks. It also supports camera controls
//  such as switching between front and back cameras, zooming, and capturing still photos.

import AVFoundation
import CoreVideo
import UIKit
import Vision

/// Protocol for receiving video capture frame processing results.
@MainActor
protocol VideoCaptureDelegate: AnyObject {
  func onPredict(result: YOLOResult)
  func onInferenceTime(speed: Double, fps: Double)
}

func bestCaptureDevice(position: AVCaptureDevice.Position) -> AVCaptureDevice {
  // print("USE TELEPHOTO: ")
  // print(UserDefaults.standard.bool(forKey: "use_telephoto"))

  if UserDefaults.standard.bool(forKey: "use_telephoto"),
    let device = AVCaptureDevice.default(.builtInTelephotoCamera, for: .video, position: position)
  {
    return device
  } else if let device = AVCaptureDevice.default(
    .builtInDualCamera, for: .video, position: position)
  {
    return device
  } else if let device = AVCaptureDevice.default(
    .builtInWideAngleCamera, for: .video, position: position)
  {
    return device
  } else {
    fatalError("Missing expected back camera device.")
  }
}

class VideoCapture: NSObject, @unchecked Sendable {
  var predictor: Predictor!
  var previewLayer: AVCaptureVideoPreviewLayer?
  weak var delegate: VideoCaptureDelegate?
  var captureDevice: AVCaptureDevice?
  let captureSession = AVCaptureSession()
  var videoInput: AVCaptureDeviceInput? = nil
  let videoOutput = AVCaptureVideoDataOutput()
  var photoOutput = AVCapturePhotoOutput()
  let cameraQueue = DispatchQueue(label: "camera-queue")
  var lastCapturedPhoto: UIImage? = nil
  var inferenceOK = true
  var longSide: CGFloat = 3
  var shortSide: CGFloat = 4
  var frameSizeCaptured = false

  private var currentBuffer: CVPixelBuffer?
  
  // MARK: - 비디오 녹화 관련 프로퍼티
  let movieFileOutput = AVCaptureMovieFileOutput()
  var isRecording = false
  var currentRecordingURL: URL?
  var recordingCompletionHandler: ((URL?, Error?) -> Void)?
  var currentPosition: AVCaptureDevice.Position = .back
  var currentZoomFactor: CGFloat = 1.0
  var audioEnabled = true
  
  // MARK: - 프레임 레이트 및 슬로우 모션 관련 프로퍼티
  var currentFrameRate: Int = 30
  var isSlowMotionEnabled: Bool = false
  var currentDevice: AVCaptureDevice? {
    return captureDevice
  }

  func setUp(
    sessionPreset: AVCaptureSession.Preset = .hd1280x720,
    position: AVCaptureDevice.Position,
    orientation: UIDeviceOrientation,
    completion: @escaping (Bool) -> Void
  ) {
    cameraQueue.async {
      let success = self.setUpCamera(
        sessionPreset: sessionPreset, position: position, orientation: orientation)
      DispatchQueue.main.async {
        completion(success)
      }
    }
  }

  func setUpCamera(
    sessionPreset: AVCaptureSession.Preset, position: AVCaptureDevice.Position,
    orientation: UIDeviceOrientation
  ) -> Bool {
    captureSession.beginConfiguration()
    captureSession.sessionPreset = sessionPreset

    captureDevice = bestCaptureDevice(position: position)
    videoInput = try! AVCaptureDeviceInput(device: captureDevice!)

    if captureSession.canAddInput(videoInput!) {
      captureSession.addInput(videoInput!)
    }
    var videoOrientaion = AVCaptureVideoOrientation.portrait
    switch orientation {
    case .portrait:
      videoOrientaion = .portrait
    case .landscapeLeft:
      videoOrientaion = .landscapeRight
    case .landscapeRight:
      videoOrientaion = .landscapeLeft
    default:
      videoOrientaion = .portrait
    }
    let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
    previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
    previewLayer.connection?.videoOrientation = videoOrientaion
    self.previewLayer = previewLayer

    let settings: [String: Any] = [
      kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)
    ]

    videoOutput.videoSettings = settings
    videoOutput.alwaysDiscardsLateVideoFrames = true
    videoOutput.setSampleBufferDelegate(self, queue: cameraQueue)
    if captureSession.canAddOutput(videoOutput) {
      captureSession.addOutput(videoOutput)
    }
    if captureSession.canAddOutput(photoOutput) {
      captureSession.addOutput(photoOutput)
      photoOutput.isHighResolutionCaptureEnabled = true
      //            photoOutput.isLivePhotoCaptureEnabled = photoOutput.isLivePhotoCaptureSupported
    }
    
    // Add movie file output for video recording
    if captureSession.canAddOutput(movieFileOutput) {
      captureSession.addOutput(movieFileOutput)
    }
    
    // Add audio input for video recording if available
    addAudioInput()

    // We want the buffers to be in portrait orientation otherwise they are
    // rotated by 90 degrees. Need to set this _after_ addOutput()!
    // let curDeviceOrientation = UIDevice.current.orientation
    let connection = videoOutput.connection(with: AVMediaType.video)
    connection?.videoOrientation = videoOrientaion
    if position == .front {
      connection?.isVideoMirrored = true
    }

    // Configure captureDevice
    do {
      try captureDevice!.lockForConfiguration()
    } catch {
      print("device configuration not working")
    }
    // captureDevice.setFocusModeLocked(lensPosition: 1.0, completionHandler: { (time) -> Void in })
    if captureDevice!.isFocusModeSupported(AVCaptureDevice.FocusMode.continuousAutoFocus),
      captureDevice!.isFocusPointOfInterestSupported
    {
      captureDevice!.focusMode = AVCaptureDevice.FocusMode.continuousAutoFocus
      captureDevice!.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5)
    }
    captureDevice!.exposureMode = AVCaptureDevice.ExposureMode.continuousAutoExposure
    captureDevice!.unlockForConfiguration()

    captureSession.commitConfiguration()
    return true
  }

  func start() {
    if !captureSession.isRunning {
      DispatchQueue.global().async {
        self.captureSession.startRunning()
      }
    }
  }

  func stop() {
    if captureSession.isRunning {
      DispatchQueue.global().async {
        self.captureSession.stopRunning()
      }
    }
  }

  func setZoomRatio(ratio: CGFloat) {
    do {
      try captureDevice!.lockForConfiguration()
      defer {
        captureDevice!.unlockForConfiguration()
      }
      captureDevice!.videoZoomFactor = ratio
    } catch {}
  }

  private func predictOnFrame(sampleBuffer: CMSampleBuffer) {
    guard let predictor = predictor else {
      print("predictor is nil")
      return
    }
    if currentBuffer == nil, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
      currentBuffer = pixelBuffer
      if !frameSizeCaptured {
        let frameWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let frameHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        longSide = max(frameWidth, frameHeight)
        shortSide = min(frameWidth, frameHeight)
        frameSizeCaptured = true
      }

      /// - Tag: MappingOrientation
      // The frame is always oriented based on the camera sensor,
      // so in most cases Vision needs to rotate it for the model to work as expected.
      var imageOrientation: CGImagePropertyOrientation = .up
      //            switch UIDevice.current.orientation {
      //            case .portrait:
      //                imageOrientation = .up
      //            case .portraitUpsideDown:
      //                imageOrientation = .down
      //            case .landscapeLeft:
      //                imageOrientation = .up
      //            case .landscapeRight:
      //                imageOrientation = .up
      //            case .unknown:
      //                imageOrientation = .up
      //
      //            default:
      //                imageOrientation = .up
      //            }

      predictor.predict(sampleBuffer: sampleBuffer, onResultsListener: self, onInferenceTime: self)
      currentBuffer = nil
    }
  }

  func updateVideoOrientation(orientation: AVCaptureVideoOrientation) {
    guard let connection = videoOutput.connection(with: .video) else { return }

    connection.videoOrientation = orientation
    let currentInput = self.captureSession.inputs.first as? AVCaptureDeviceInput
    if currentInput?.device.position == .front {
      connection.isVideoMirrored = true
    } else {
      connection.isVideoMirrored = false
    }
    let o = connection.videoOrientation
    self.previewLayer?.connection?.videoOrientation = connection.videoOrientation
  }

  deinit {
    print("VideoCapture: deinit called - ensuring capture session is stopped")
    if captureSession.isRunning {
      captureSession.stopRunning()
    }

    // Remove all inputs and outputs
    if let inputs = captureSession.inputs as? [AVCaptureInput] {
      for input in inputs {
        captureSession.removeInput(input)
      }
    }

    if let outputs = captureSession.outputs as? [AVCaptureOutput] {
      for output in outputs {
        captureSession.removeOutput(output)
      }
    }

    print("VideoCapture: deinit completed")
  }
  
  // MARK: - Video Recording Functions
  
  /// Simple recording state check
  func getCurrentRecordingState() -> Bool {
    return movieFileOutput.isRecording
  }
  
  /// Get current recording file path
  func getCurrentRecordingPath() -> String? {
    return currentRecordingURL?.path
  }
  
  func startRecording(completion: @escaping (URL?, Error?) -> Void) {
    // 이미 녹화 중인지 실제 movieFileOutput 상태로 확인
    if movieFileOutput.isRecording {
      completion(nil, NSError(domain: "VideoCapture", code: 100, userInfo: [NSLocalizedDescriptionKey: "이미 녹화 중입니다"]))
      return
    }
    
    // isRecording 플래그가 true인데 실제로 녹화가 진행 중이 아닌 경우
    if isRecording && !movieFileOutput.isRecording {
      print("DEBUG: 상태 불일치 감지 - isRecording은 true이나 실제로는 녹화 중이 아님")
      isRecording = false // 상태 재설정
    }
    
    // 고유한 파일 이름 생성: 타임스탬프 + UUID
    let timestamp = Date().timeIntervalSince1970
    let uuid = UUID().uuidString.prefix(8)
    let fileName = "recording_\(timestamp)_\(uuid).mp4"
    
    // Documents 디렉토리에 저장 (앱이 접근 가능한 디렉토리)
    let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    let fileURL = documentsPath.appendingPathComponent(fileName)
    
    // 파일이 이미 존재하면 삭제
    try? FileManager.default.removeItem(at: fileURL)

    cameraQueue.async { [weak self] in
      guard let self = self else { 
        DispatchQueue.main.async { completion(nil, NSError(domain: "VideoCapture", code: 105, userInfo: [NSLocalizedDescriptionKey: "VideoCapture 객체가 해제됨"])) }
        return 
      }
      
      // captureSession이 실행 중인지 확인
      guard self.captureSession.isRunning else {
        DispatchQueue.main.async { completion(nil, NSError(domain: "VideoCapture", code: 107, userInfo: [NSLocalizedDescriptionKey: "카메라 세션이 실행 중이 아님"])) }
        return
      }
      
      // 출력이 모두 설정되어 있는지 확인
      if !self.captureSession.outputs.contains(self.movieFileOutput) {
        // 출력이 없으면 다시 추가 시도
        self.captureSession.beginConfiguration()
        if self.captureSession.canAddOutput(self.movieFileOutput) {
          self.captureSession.addOutput(self.movieFileOutput)
          print("DEBUG: movieFileOutput 다시 추가됨")
        }
        self.captureSession.commitConfiguration()
        
        // 여전히 없으면 오류 반환
        if !self.captureSession.outputs.contains(self.movieFileOutput) {
          DispatchQueue.main.async { completion(nil, NSError(domain: "VideoCapture", code: 110, userInfo: [NSLocalizedDescriptionKey: "movieFileOutput을 세션에 추가할 수 없음"])) }
          return
        }
      }
      
      // 실제 녹화 시작 전에 플래그 설정
      self.isRecording = true
      
      if self.movieFileOutput.isRecording == false {
        // 비디오 설정 구성
        if let connection = self.movieFileOutput.connection(with: .video) {
          // 비디오 방향 설정
          connection.videoOrientation = .portrait
          if let currentInput = self.captureSession.inputs.first as? AVCaptureDeviceInput {
            connection.isVideoMirrored = currentInput.device.position == .front
          }
          
          // 비디오 안정화 설정 (가능한 경우)
          if connection.isVideoStabilizationSupported {
            connection.preferredVideoStabilizationMode = .auto
          }
        }
        
        self.recordingCompletionHandler = completion
        self.currentRecordingURL = fileURL
        
        // 녹화 시작 시도
        print("DEBUG: 녹화 시작 시도 to \(fileURL.path)")
        self.movieFileOutput.startRecording(to: fileURL, recordingDelegate: self)
        print("DEBUG: Video recording started successfully")
        
        // 즉시 성공 응답 (delegate에서 실제 상태 처리)
        DispatchQueue.main.async {
          completion(fileURL, nil)
        }
      } else {
        self.isRecording = false
        DispatchQueue.main.async {
          completion(nil, NSError(domain: "VideoCapture", code: 101, userInfo: [NSLocalizedDescriptionKey: "녹화 시작 실패 - 이미 다른 녹화가 진행 중"]))
        }
      }
    }
  }
  
  func stopRecording(completion: @escaping (URL?, Error?) -> Void) {
    // 실제 녹화 상태 확인 (이중 검증)
    if !movieFileOutput.isRecording {
      // 상태 불일치 감지 - isRecording 플래그 재설정
      if isRecording {
        print("DEBUG: 상태 불일치 감지 - isRecording은 true이나 실제로는 녹화 중이 아님")
        isRecording = false
      }
      
      completion(nil, NSError(domain: "VideoCapture", code: 102, userInfo: [NSLocalizedDescriptionKey: "녹화 중이 아닙니다"]))
      return
    }
    
    cameraQueue.async { [weak self] in
      guard let self = self else {
        DispatchQueue.main.async { completion(nil, NSError(domain: "VideoCapture", code: 108, userInfo: [NSLocalizedDescriptionKey: "VideoCapture 객체가 해제됨"])) }
        return
      }
      
      // 녹화 중인지 다시 확인 (비동기 작업 중 상태가 변경되었을 수 있음)
      if self.movieFileOutput.isRecording {
        print("DEBUG: 녹화 중지 시도 중...")
        
        // 현재 녹화 URL 저장
        let recordingURL = self.currentRecordingURL
        
        // 녹화 중지
        self.movieFileOutput.stopRecording()
        
        // 상태 업데이트
        self.isRecording = false
        
        // 즉시 현재 파일 경로 반환
        DispatchQueue.main.async {
          if let url = recordingURL {
            print("DEBUG: 녹화 중지 완료 - 파일 경로: \(url.path)")
            completion(url, nil)
          } else {
            print("DEBUG: 녹화 중지되었으나 URL이 없음")
            completion(nil, NSError(domain: "VideoCapture", code: 109, userInfo: [NSLocalizedDescriptionKey: "녹화 URL을 찾을 수 없음"]))
          }
        }
      } else {
        print("DEBUG: ⚠️ 상태 불일치: stopRecording 호출됨 - 실제 녹화 중이 아님")
        self.isRecording = false
        
        DispatchQueue.main.async {
          completion(nil, NSError(domain: "VideoCapture", code: 103, userInfo: [NSLocalizedDescriptionKey: "녹화가 이미 중지됨"]))
        }
      }
    }
  }
  
  // MARK: - Frame Rate Management Functions
  func getSupportedFrameRatesInfo() -> [String: Bool] {
    let fpsValues = [30.0, 60.0, 90.0, 120.0]
    var result = [String: Bool]()
    
    for fps in fpsValues {
      let key = "\(Int(fps))fps"
      result[key] = isFrameRateSupported(fps)
    }
    
    print("DEBUG: Supported frame rates: \(result)")
    return result
  }

  func isFrameRateSupported(_ fps: Double) -> Bool {
    guard let device = self.currentDevice else { return false }
    
    // 모든 포맷에서 확인
    for format in device.formats {
      for range in format.videoSupportedFrameRateRanges {
        if fps >= range.minFrameRate && fps <= range.maxFrameRate {
          return true
        }
      }
    }
    return false
  }

  // 특정 FPS를 지원하는 최적의 포맷 찾기
  private func findFormatSupportingFrameRate(_ fps: Double) -> AVCaptureDevice.Format? {
    guard let device = self.currentDevice else { return nil }
    
    // 현재 해상도 가져오기
    let currentDimensions = CMVideoFormatDescriptionGetDimensions(device.activeFormat.formatDescription)
    let currentResolution = currentDimensions.width * currentDimensions.height
    
    var bestFormat: AVCaptureDevice.Format? = nil
    var bestResolutionMatch: Int = Int.max
    
    for format in device.formats {
      // 이 포맷이 원하는 fps를 지원하는지 확인
      let ranges = format.videoSupportedFrameRateRanges
      let supportsFrameRate = ranges.contains { range in
        return fps >= range.minFrameRate && fps <= range.maxFrameRate
      }
      
      if supportsFrameRate {
        let formatDimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        let formatResolution = formatDimensions.width * formatDimensions.height
        let resolutionDiff = abs(Int(formatResolution) - Int(currentResolution))
        
        // 이전에 찾은 포맷보다 현재 해상도에 더 가까운 포맷인 경우 업데이트
        if bestFormat == nil || resolutionDiff < bestResolutionMatch {
          bestFormat = format
          bestResolutionMatch = resolutionDiff
        }
      }
    }
    
    return bestFormat
  }
  
  func setFrameRate(_ fps: Int) -> Bool {
    guard let device = self.currentDevice else { 
      print("DEBUG: Cannot set frame rate - no device available")
      return false 
    }
    
    // 이미 같은 FPS라면 변경 불필요
    if self.currentFrameRate == fps {
      print("DEBUG: Frame rate already set to \(fps) FPS")
      return true
    }
    
    // 먼저 현재 포맷이 이 FPS를 지원하는지 확인
    var currentFormatSupported = false
    for range in device.activeFormat.videoSupportedFrameRateRanges {
      if Double(fps) >= range.minFrameRate && Double(fps) <= range.maxFrameRate {
        currentFormatSupported = true
        break
      }
    }
    
    // 현재 포맷이 지원하지 않는 경우, 지원하는 포맷을 찾음
    if !currentFormatSupported {
      print("DEBUG: Current format does not support \(fps) FPS, searching for compatible format...")
      
      guard let newFormat = findFormatSupportingFrameRate(Double(fps)) else {
        print("DEBUG: No format found supporting \(fps) FPS")
        return false
      }
      
      // 새 포맷으로 전환
      let originalFormat = device.activeFormat // 원래 포맷 저장
      do {
        try device.lockForConfiguration()
        device.activeFormat = newFormat
        device.unlockForConfiguration()
        
        let dimensions = CMVideoFormatDescriptionGetDimensions(newFormat.formatDescription)
        print("DEBUG: Switched to format with resolution \(dimensions.width)x\(dimensions.height) supporting \(fps) FPS")
      } catch {
        print("DEBUG: Failed to switch format: \(error)")
        // 실패한 경우 원래 포맷으로 복원
        do {
          try device.lockForConfiguration()
          device.activeFormat = originalFormat
          device.unlockForConfiguration()
          print("DEBUG: Restored original format after failure")
        } catch {
          print("DEBUG: Failed to restore original format: \(error)")
        }
        return false
      }
    }
    
    // 이제 FPS를 설정
    do {
      try device.lockForConfiguration()
      
      // 30프레임 디바이스에서 그 이상을 요청한 경우 최대 프레임레이트로 제한
      var targetFps = fps
      let maxSupportedFps = Int(device.activeFormat.videoSupportedFrameRateRanges.map { $0.maxFrameRate }.max() ?? 30.0)
      
      if targetFps > maxSupportedFps {
        print("DEBUG: Requested \(fps) FPS, but device only supports up to \(maxSupportedFps) FPS. Using \(maxSupportedFps) FPS instead.")
        targetFps = maxSupportedFps
      }
      
      let duration = CMTime(value: 1, timescale: CMTimeScale(targetFps))
      device.activeVideoMinFrameDuration = duration
      device.activeVideoMaxFrameDuration = duration
      self.currentFrameRate = targetFps
      
      device.unlockForConfiguration()
      print("DEBUG: Frame rate successfully set to \(targetFps) FPS")
      return true
    } catch {
      print("DEBUG: Failed to set frame rate: \(error)")
      return false
    }
  }
  
  // MARK: - Slow Motion Functions
  func isSlowMotionSupported() -> Bool {
    guard let device = currentDevice else { return false }
    
    // 120fps 이상을 지원하는 포맷이 있는지 확인
    for format in device.formats {
      for range in format.videoSupportedFrameRateRanges {
        if range.maxFrameRate >= 120 {
          return true
        }
      }
    }
    return false
  }
  
  func getMaxSlowMotionFrameRate() -> Int {
    guard let device = currentDevice else { return 30 }
    
    var maxFrameRate: Double = 30
    for format in device.formats {
      for range in format.videoSupportedFrameRateRanges {
        if range.maxFrameRate > maxFrameRate {
          maxFrameRate = range.maxFrameRate
        }
      }
    }
    
    return Int(maxFrameRate)
  }
  
  func enableSlowMotion(_ enable: Bool) -> Bool {
    guard let device = currentDevice else { return false }
    
    // 이미 원하는 상태면 변경 필요 없음
    if isSlowMotionEnabled == enable {
      print("DEBUG: 슬로우 모션 상태가 이미 \(enable ? "활성화" : "비활성화") 되어있습니다.")
      return true
    }
    
    // 녹화 중에는 모드 변경 금지
    if isRecording {
      print("DEBUG: ⚠️ 녹화 중에는 슬로우 모션 모드를 변경할 수 없습니다")
      return false
    }
    
    do {
      try device.lockForConfiguration()
      
      if enable {
        // 슬로우 모션 활성화 - 120fps 또는 240fps로 설정
        let targetFps = min(240, getMaxSlowMotionFrameRate())
        let duration = CMTime(value: 1, timescale: CMTimeScale(targetFps))
        device.activeVideoMinFrameDuration = duration
        device.activeVideoMaxFrameDuration = duration
        
        currentFrameRate = targetFps
        isSlowMotionEnabled = true
        
        print("DEBUG: ✅ 슬로우 모션 활성화 성공: \(targetFps) FPS")
      } else {
        // 일반 모드로 복귀 - 30fps로 설정
        let duration = CMTime(value: 1, timescale: 30)
        device.activeVideoMinFrameDuration = duration
        device.activeVideoMaxFrameDuration = duration
        
        currentFrameRate = 30
        isSlowMotionEnabled = false
        
        print("DEBUG: ✅ 일반 모드로 복귀 성공: 30 FPS")
      }
      
      device.unlockForConfiguration()
      return true
    } catch {
      print("DEBUG: 슬로우 모션 모드 변경 실패: \(error)")
      return false
    }
  }
  
  func isSlowMotionActive() -> Bool {
    return isSlowMotionEnabled && currentFrameRate >= 120
  }
  
  // MARK: - Audio Input Management
  
  /// Adds audio input to the capture session for video recording
  private func addAudioInput() {
    // Check if audio input already exists
    if hasAudioInput() {
      print("DEBUG: Audio input already exists")
      return
    }
    
    // Only add audio if enabled
    guard audioEnabled else {
      print("DEBUG: Audio disabled, skipping audio input")
      return
    }
    
    guard let audioDevice = AVCaptureDevice.default(for: .audio) else {
      print("DEBUG: No audio device available")
      return
    }
    
    do {
      let audioInput = try AVCaptureDeviceInput(device: audioDevice)
      
      captureSession.beginConfiguration()
      if captureSession.canAddInput(audioInput) {
        captureSession.addInput(audioInput)
        print("DEBUG: Audio input added successfully")
      } else {
        print("DEBUG: Cannot add audio input to session")
      }
      captureSession.commitConfiguration()
    } catch {
      print("DEBUG: Failed to create audio input: \(error)")
    }
  }
  
  /// Checks if audio input exists in the capture session
  private func hasAudioInput() -> Bool {
    return captureSession.inputs.contains { input in
      guard let deviceInput = input as? AVCaptureDeviceInput else { return false }
      return deviceInput.device.hasMediaType(.audio)
    }
  }
  
  /// Gets current recording status
  public func getRecordingStatus() -> [String: Any] {
    return [
      "isRecording": isRecording,
      "isActuallyRecording": movieFileOutput.isRecording,
      "currentRecordingURL": currentRecordingURL?.path ?? "",
      "hasMovieOutput": captureSession.outputs.contains(movieFileOutput),
      "hasAudioInput": hasAudioInput()
    ]
  }
}

extension VideoCapture: AVCaptureVideoDataOutputSampleBufferDelegate {
  func captureOutput(
    _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    guard inferenceOK else { return }
    predictOnFrame(sampleBuffer: sampleBuffer)
  }
}

extension VideoCapture: AVCapturePhotoCaptureDelegate {
  @available(iOS 11.0, *)
  func photoOutput(
    _ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?
  ) {
    guard let data = photo.fileDataRepresentation(),
      let image = UIImage(data: data)
    else {
      return
    }

    self.lastCapturedPhoto = image
  }
}

extension VideoCapture: ResultsListener, InferenceTimeListener {
  func on(inferenceTime: Double, fpsRate: Double) {
    DispatchQueue.main.async {
      self.delegate?.onInferenceTime(speed: inferenceTime, fps: fpsRate)
    }
  }

  func on(result: YOLOResult) {
    DispatchQueue.main.async {
      self.delegate?.onPredict(result: result)
    }
  }
}

extension VideoCapture: AVCaptureFileOutputRecordingDelegate {
  func fileOutput(_ output: AVCaptureFileOutput, didStartRecordingTo fileURL: URL, from connections: [AVCaptureConnection]) {
    print("DEBUG: 녹화 시작됨 at \(fileURL.path)")
    // 녹화 URL 저장
    self.currentRecordingURL = fileURL
  }
  
  func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo outputFileURL: URL, from connections: [AVCaptureConnection], error: Error?) {
    print("DEBUG: 녹화 완료됨 at \(outputFileURL.path)")
    
    // 상태 정리
    DispatchQueue.main.async {
      self.isRecording = false
      self.currentRecordingURL = nil
      self.recordingCompletionHandler = nil
    }
    
    if let error = error {
      print("DEBUG: 녹화 완료 시 오류 발생: \(error)")
    } else {
      print("DEBUG: 녹화 성공적으로 완료 - 파일 저장됨: \(outputFileURL.path)")
    }
  }
}
