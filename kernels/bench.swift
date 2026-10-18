// Host harness for matmul.metal: compiles it at runtime (Metal 4), checks every kernel bit-exactly on
// sampled outputs against a CPU reference, then times it with GPU timestamps.
//
// Usage: bench [key=value ...]
//   M N K            problem size (default 4096 each)
//   TM TN NSG        tile shape and simdgroups per threadgroup (default 128 64 4)
//   GROUP_M          grouped tile order (default 1 = row-major order)
//   BK               K chunk for the explicit K loop (default 0 = single matmul2d call)
//   kernels          comma-separated subset (default: all)
//   reps             interleaved timed repetitions; medians reported (default 15)
//   warm             seconds of warm-up before timing (default 1)
//   csv              1 = print one CSV row per kernel instead of the human-readable table
import Foundation
import Metal

var opt: [String: String] = [:]
for a in CommandLine.arguments.dropFirst() {
    let kv = a.split(separator: "=", maxSplits: 1).map(String.init)
    guard kv.count == 2 else { print("bad argument \(a); expected key=value"); exit(2) }
    opt[kv[0]] = kv[1]
}
func int(_ k: String, _ d: Int) -> Int { opt[k].flatMap { Int($0) } ?? d }
let (M, N, K) = (int("M", 4096), int("N", 4096), int("K", 4096))
let (TM, TN, NSG, GROUP_M) = (int("TM", 128), int("TN", 64), int("NSG", 4), int("GROUP_M", 1))
let BK = int("BK", 0)
let warm = Double(opt["warm"] ?? "1") ?? 1
let reps = int("reps", 15), csv = int("csv", 0) == 1
guard M % TM == 0, N % TN == 0, BK == 0 || K % BK == 0 else { print("M, N, K must be multiples of TM, TN, BK"); exit(2) }

let device = MTLCreateSystemDefaultDevice()!
let srcURL = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("matmul.metal")
let opts = MTLCompileOptions()
opts.languageVersion = .version4_0
opts.preprocessorMacros = ["TM": TM, "TN": TN, "NSG": NSG, "GROUP_M": GROUP_M, "BK": BK]
    .mapValues { NSNumber(value: $0) }
let lib: MTLLibrary
do { lib = try device.makeLibrary(source: try String(contentsOf: srcURL, encoding: .utf8), options: opts) } catch {
    print("compile failed:\n\(error)"); exit(1)
}
let queue = device.makeCommandQueue()!
if !csv {
    print("\(device.name)  M=\(M) N=\(N) K=\(K)  tile \(TM)x\(TN)x\(NSG)sg  GROUP_M=\(GROUP_M)  BK=\(BK)")
}

// Small integers so FP16, INT8 and INT4 products are all exact and comparable.
var rng = SystemRandomNumberGenerator()
let aInt = (0..<M * K).map { _ in Int8.random(in: -8...7, using: &rng) }
let bInt = (0..<K * N).map { _ in Int8.random(in: -8...7, using: &rng) }  // logical K x N
let bDev = (0..<N * K).map { bInt[($0 % K) * N + $0 / K] }  // device layout N x K

func buffer<T>(_ a: [T]) -> MTLBuffer {
    a.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
}
let aI8 = buffer(aInt), bI8 = buffer(bDev)
let aF16 = buffer(aInt.map { Float16($0) }), bF16 = buffer(bDev.map { Float16($0) })
let bI4 = buffer(stride(from: 0, to: bDev.count, by: 2).map {
    UInt8(UInt8(bitPattern: bDev[$0]) & 0xF) | (UInt8(bitPattern: bDev[$0 + 1]) << 4)
})
let cBuf = device.makeBuffer(length: M * N * 4, options: .storageModeShared)!
var dims = SIMD3<UInt32>(UInt32(M), UInt32(N), UInt32(K))

func run(_ pso: MTLComputePipelineState, _ a: MTLBuffer, _ b: MTLBuffer, iters: Int) -> Double {
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    enc.setBuffers([a, b, cBuf], offsets: [0, 0, 0], range: 0..<3)
    enc.setBytes(&dims, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 3)
    for _ in 0..<iters {
        enc.dispatchThreadgroups(MTLSize(width: N / TN, height: M / TM, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32 * NSG, height: 1, depth: 1))
    }
    enc.endEncoding()
    cb.commit(); cb.waitUntilCompleted()
    if let e = cb.error { print("GPU error: \(e)"); exit(1) }
    return (cb.gpuEndTime - cb.gpuStartTime) / Double(iters)
}

func maxError(_ read: (Int) -> Double) -> Double {
    var maxErr = 0.0
    for _ in 0..<256 {
        let i = Int.random(in: 0..<M), j = Int.random(in: 0..<N)
        var ref = 0
        for k in 0..<K { ref += Int(aInt[i * K + k]) * Int(bInt[k * N + j]) }
        maxErr = max(maxErr, abs(read(i * N + j) - Double(ref)))
    }
    return maxErr
}

func out<T: BinaryFloatingPoint>(_: T.Type) -> (Int) -> Double { { Double(cBuf.contents().assumingMemoryBound(to: T.self)[$0]) } }
let outI32: (Int) -> Double = { Double(cBuf.contents().assumingMemoryBound(to: Int32.self)[$0]) }
let all: [(String, MTLBuffer, MTLBuffer, String, (Int) -> Double)] = [
    ("mm_f16_f32", aF16, bF16, "FP16 (FP32 out)", out(Float.self)),
    ("mm_f16_f16", aF16, bF16, "FP16 (FP16 out)", out(Float16.self)),
    ("mm_i8_i32", aI8, bI8, "W8A8 INT8xINT8", outI32),
    ("mm_f16xi8_f32", aF16, bI8, "W8A16 FP16xINT8", out(Float.self)),
    ("mm_i8xi4_i32", aI8, bI4, "W4A8 INT8xINT4", outI32),
    ("mm_f16xi4_f32", aF16, bI4, "W4A16 FP16xINT4", out(Float.self)),
]
let want = opt["kernels"].map { Set($0.split(separator: ",").map(String.init)) }
let flops = 2.0 * Double(M) * Double(N) * Double(K)
let chosen = all.filter { want?.contains($0.0) ?? true }
let psos = chosen.map { try! device.makeComputePipelineState(function: lib.makeFunction(name: $0.0)!) }

// Correctness first. FP16 output can't hold the exact sums, so only its speed is reported.
var errs: [Double] = []
for (i, k) in chosen.enumerated() {
    _ = run(psos[i], k.1, k.2, iters: 1)
    errs.append(k.0 == "mm_f16_f16" ? .nan : maxError(k.4))
}

// Ramp GPU clocks for ~1 s, then interleave: every rep times each kernel back-to-back so clock or
// background-load drift affects all kernels alike.
let warmStart = Date()
while Date().timeIntervalSince(warmStart) < warm { for (i, k) in chosen.enumerated() { _ = run(psos[i], k.1, k.2, iters: 3) } }
let iters = max(3, Int(2e12 / flops))
var times = Array(repeating: [Double](), count: chosen.count)
for _ in 0..<reps {
    for (i, k) in chosen.enumerated() { times[i].append(run(psos[i], k.1, k.2, iters: iters)) }
}
func median(_ x: [Double]) -> Double { let s = x.sorted(); return s[s.count / 2] }
let baseIdx = chosen.firstIndex { $0.0 == "mm_f16_f32" }

for (i, k) in chosen.enumerated() {
    let med = median(times[i]), best = times[i].min()!
    // Per-rep speedup over FP16 measured in the same rep, then the median across reps.
    let ratio = baseIdx.map { b in median(zip(times[b], times[i]).map { $0 / $1 }) } ?? .nan
    if csv {
        print([M, N, K, TM, TN, NSG, GROUP_M, BK].map(String.init).joined(separator: ",") +
              ",\(k.0),\(med * 1e3),\(best * 1e3),\(flops / med / 1e12),\(ratio),\(errs[i])")
    } else {
        print(String(format: "  %-17@ median %8.3f ms  %6.1f T(FL)OPS  (best %6.1f)  vs FP16 %5.2fx  max err %@",
                     k.3 as NSString, med * 1e3, flops / med / 1e12, flops / best / 1e12, ratio,
                     errs[i].isNaN ? "n/a" : String(errs[i])))
    }
}
