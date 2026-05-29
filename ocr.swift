import Vision
import AppKit
import Foundation

let stderr = FileHandle.standardError

guard CommandLine.arguments.count > 1 else {
    stderr.write(Data("Usage: swift ocr.swift <image_path>\n".utf8))
    exit(1)
}

let imagePath = CommandLine.arguments[1]

guard let data = try? Data(contentsOf: URL(fileURLWithPath: imagePath)),
      let image = NSImage(data: data),
      let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    stderr.write(Data("Error: Cannot load image from \(imagePath)\n".utf8))
    exit(1)
}

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.recognitionLanguages = ["zh-Hans", "zh-Hant", "en"]
request.usesLanguageCorrection = true
request.automaticallyDetectsLanguage = true

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])

do {
    try handler.perform([request])
} catch {
    stderr.write(Data("Error: \(error.localizedDescription)\n".utf8))
    exit(1)
}

guard let observations = request.results else {
    exit(0)
}

// 按位置排序（从上到下，从左到右）
let sorted = observations.sorted { a, b in
    let ay = a.boundingBox.origin.y
    let by = b.boundingBox.origin.y
    if abs(ay - by) > 0.01 {
        return ay > by
    }
    return a.boundingBox.origin.x < b.boundingBox.origin.x
}

for obs in sorted {
    if let candidate = obs.topCandidates(1).first {
        print(candidate.string)
    }
}
