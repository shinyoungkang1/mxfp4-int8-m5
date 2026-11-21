// Host harness for matmul_fp.metal: checks and times macOS 27.2 FP4/FP8 matmul2d kernels alongside
// FP16, INT8xINT8 and INT8xINT4 anchors on the M5 Neural Accelerators.
//
// Usage: bench_fp [key=value ...]
//   M N K            problem size (default 4096 each)
//   TM TN NSG        tile shape and simdgroups per threadgroup (default 128 64 4)
//   GROUP_M          grouped tile order (default 8)
//   BK               K chunk for explicit K loop (default 0 = single matmul2d call)
//   scaled           1 = also compile/run block-scaled MXFP4 kernels (default 0)
//   lang             Metal language version: 4.1 (default), 4.0, or default
//   kernels          comma-separated subset (default: all available)
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
let (TM, TN, NSG, GROUP_M) = (int("TM", 128), int("TN", 64), int("NSG", 4), int("GROUP_M", 8))
let BK = int("BK", 0)
let scaled = int("scaled", 0)
let lang = opt["lang"] ?? "4.1"
let warm = Double(opt["warm"] ?? "1") ?? 1
let reps = int("reps", 15), csv = int("csv", 0) == 1
guard M % TM == 0, N % TN == 0, K % 32 == 0, BK == 0 || K % BK == 0 else {
    print("M, N, K must be multiples of TM, TN, 32, BK"); exit(2)
}

let device = MTLCreateSystemDefaultDevice()!
let srcURL = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("matmul_fp.metal")
let opts = MTLCompileOptions()
if lang == "4.1" {
    opts.languageVersion = MTLLanguageVersion(rawValue: (4 << 16) + 1)!
} else if lang == "4.0" {
    opts.languageVersion = .version4_0
}
opts.preprocessorMacros = ["TM": TM, "TN": TN, "NSG": NSG, "GROUP_M": GROUP_M, "BK": BK, "ENABLE_SCALED": scaled]
    .mapValues { NSNumber(value: $0) }
let lib: MTLLibrary
do { lib = try device.makeLibrary(source: try String(contentsOf: srcURL, encoding: .utf8), options: opts) } catch {
    print("compile failed (lang=\(lang), scaled=\(scaled)):\n\(error)"); exit(1)
}
let queue = device.makeCommandQueue()!
if !csv {
    print("\(device.name)  M=\(M) N=\(N) K=\(K)  tile \(TM)x\(TN)x\(NSG)sg  GROUP_M=\(GROUP_M)  BK=\(BK)  lang=\(lang)")
}

// Values in [-4, 4] are all exact in E2M1, E4M3, E5M2, INT4, INT8 and FP16, and K <= 14336 sums stay < 2^24.
var rng = SystemRandomNumberGenerator()
let aInt = (0..<M * K).map { _ in Int8.random(in: -4...4, using: &rng) }
let bInt = (0..<K * N).map { _ in Int8.random(in: -4...4, using: &rng) }  // logical K x N
let bDev = (0..<N * K).map { bInt[($0 % K) * N + $0 / K] }  // device layout N x K

func packNibble(_ arr: [Int8], encode: (Int8) -> UInt8) -> [UInt8] {
    stride(from: 0, to: arr.count, by: 2).map {
        (encode(arr[$0]) & 0xF) | ((encode(arr[$0 + 1]) & 0xF) << 4)
    }
}
func encE2M1(_ v: Int8) -> UInt8 {
    let s: UInt8 = v < 0 ? 0x8 : 0x0
    let mag: UInt8
    switch abs(Int(v)) {
    case 0: mag = 0
    case 1: mag = 2  // 2^(1-1)*1.0
    case 2: mag = 4  // 2^(2-1)*1.0
    case 3: mag = 5  // 2^(2-1)*1.5
    case 4: mag = 6  // 2^(3-1)*1.0
    default: fatalError("unsupported value \(v)")
    }
    return s | mag
}
func encE4M3(_ v: Int8) -> UInt8 {
    let s: UInt8 = v < 0 ? 0x80 : 0x00
    let mag: UInt8
    switch abs(Int(v)) {
    case 0: mag = 0x00
    case 1: mag = 0x38
    case 2: mag = 0x40
    case 3: mag = 0x44
    case 4: mag = 0x48
    default: fatalError("unsupported value \(v)")
    }
    return s | mag
}
func encE5M2(_ v: Int8) -> UInt8 {
    let s: UInt8 = v < 0 ? 0x80 : 0x00
    let mag: UInt8
    switch abs(Int(v)) {
    case 0: mag = 0x00
    case 1: mag = 0x3C
    case 2: mag = 0x40
    case 3: mag = 0x42
    case 4: mag = 0x44
    default: fatalError("unsupported value \(v)")
    }
    return s | mag
}

func buffer<T>(_ a: [T]) -> MTLBuffer {
    a.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
}
let aI8 = buffer(aInt), bI8 = buffer(bDev)
let aF16 = buffer(aInt.map { Float16($0) }), bF16 = buffer(bDev.map { Float16($0) })
let bI4 = buffer(packNibble(bDev) { UInt8(bitPattern: $0) & 0xF })
let aFP4 = buffer(packNibble(aInt, encode: encE2M1))
let bFP4 = buffer(packNibble(bDev, encode: encE2M1))
let aFP8E4 = buffer(aInt.map(encE4M3)), bFP8E4 = buffer(bDev.map(encE4M3))
let aFP8E5 = buffer(aInt.map(encE5M2)), bFP8E5 = buffer(bDev.map(encE5M2))
let sA_raw = (0..<(M * (K / 32))).map { _ in UInt8.random(in: 0x7E...0x80, using: &rng) }
let sB_raw = (0..<(N * (K / 32))).map { _ in UInt8.random(in: 0x7E...0x80, using: &rng) }  // N x (K/32)
let sA = buffer(sA_raw)
let sB = buffer(sB_raw)
let cBuf = device.makeBuffer(length: M * N * 4, options: .storageModeShared)!
var dims = SIMD3<UInt32>(UInt32(M), UInt32(N), UInt32(K))

struct Spec {
    let fn: String
    let a: MTLBuffer
    let b: MTLBuffer
    let sa: MTLBuffer?
    let sb: MTLBuffer?
    let label: String
    let exact: Bool
    let read: (Int) -> Double
}

func run(_ pso: MTLComputePipelineState, _ s: Spec, iters: Int) -> Double {
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    let bufs: [MTLBuffer?] = [s.a, s.b, cBuf]
    let offs = [0, 0, 0]
    enc.setBuffers(bufs, offsets: offs, range: 0..<3)
    enc.setBytes(&dims, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 3)
    if let sb = s.sb { enc.setBuffer(sb, offset: 0, index: 4) }
    if let sa = s.sa { enc.setBuffer(sa, offset: 0, index: 5) }
    for _ in 0..<iters {
        enc.dispatchThreadgroups(MTLSize(width: N / TN, height: M / TM, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32 * NSG, height: 1, depth: 1))
    }
    enc.endEncoding()
    cb.commit(); cb.waitUntilCompleted()
    if let e = cb.error { print("GPU error on \(s.fn): \(e)"); exit(1) }
    return (cb.gpuEndTime - cb.gpuStartTime) / Double(iters)
}

func ue8m0(_ e: UInt8) -> Double { e == 0x7E ? 0.5 : (e == 0x7F ? 1.0 : 2.0) }

func maxError(_ s: Spec) -> Double {
    var maxErr = 0.0
    let useSA = s.sa != nil, useSB = s.sb != nil
    let kb = K / 32
    for _ in 0..<256 {
        let i = Int.random(in: 0..<M), j = Int.random(in: 0..<N)
        var ref = 0.0
        for k in 0..<K {
            let sa = useSA ? ue8m0(sA_raw[i * kb + (k >> 5)]) : 1.0
            let sb = useSB ? ue8m0(sB_raw[j * kb + (k >> 5)]) : 1.0
            ref += Double(aInt[i * K + k]) * sa * Double(bInt[k * N + j]) * sb
        }
        maxErr = max(maxErr, abs(s.read(i * N + j) - ref))
    }
    return maxErr
}

func out<T: BinaryFloatingPoint>(_: T.Type) -> (Int) -> Double { { Double(cBuf.contents().assumingMemoryBound(to: T.self)[$0]) } }
let outI32: (Int) -> Double = { Double(cBuf.contents().assumingMemoryBound(to: Int32.self)[$0]) }

var all: [Spec] = [
    Spec(fn: "mm_f16_f32",       a: aF16,   b: bF16,   sa: nil, sb: nil, label: "FP16 (FP32 out)",   exact: true,  read: out(Float.self)),
    Spec(fn: "mm_i8_i32",        a: aI8,    b: bI8,    sa: nil, sb: nil, label: "W8A8 INT8xINT8",    exact: true,  read: outI32),
    Spec(fn: "mm_i8xi4_i32",     a: aI8,    b: bI4,    sa: nil, sb: nil, label: "W4A8 INT8xINT4",    exact: true,  read: outI32),
    Spec(fn: "mm_fp4_f32",       a: aFP4,   b: bFP4,   sa: nil, sb: nil, label: "FP4xFP4 (FP32)",    exact: true,  read: out(Float.self)),
    Spec(fn: "mm_fp4_f16",       a: aFP4,   b: bFP4,   sa: nil, sb: nil, label: "FP4xFP4 (FP16)",    exact: false, read: out(Float16.self)),
    Spec(fn: "mm_f16xfp4_f32",   a: aF16,   b: bFP4,   sa: nil, sb: nil, label: "FP16xFP4 (FP32)",   exact: true,  read: out(Float.self)),
    Spec(fn: "mm_f16xfp4_f16",   a: aF16,   b: bFP4,   sa: nil, sb: nil, label: "FP16xFP4 (FP16)",   exact: false, read: out(Float16.self)),
    Spec(fn: "mm_fp8e4m3_f32",   a: aFP8E4, b: bFP8E4, sa: nil, sb: nil, label: "FP8_E4M3xE4M3 (32)",exact: true,  read: out(Float.self)),
    Spec(fn: "mm_fp8e4m3_f16",   a: aFP8E4, b: bFP8E4, sa: nil, sb: nil, label: "FP8_E4M3xE4M3 (16)",exact: false, read: out(Float16.self)),
    Spec(fn: "mm_f16xfp8e4m3_f32", a: aF16, b: bFP8E4, sa: nil, sb: nil, label: "FP16xFP8_E4M3",     exact: true,  read: out(Float.self)),
    Spec(fn: "mm_fp8e5m2_f32",   a: aFP8E5, b: bFP8E5, sa: nil, sb: nil, label: "FP8_E5M2xE5M2 (32)",exact: true,  read: out(Float.self)),
    Spec(fn: "mm_fp8e5m2_f16",   a: aFP8E5, b: bFP8E5, sa: nil, sb: nil, label: "FP8_E5M2xE5M2 (16)",exact: false, read: out(Float16.self)),
    Spec(fn: "mm_f16xfp8e5m2_f32", a: aF16, b: bFP8E5, sa: nil, sb: nil, label: "FP16xFP8_E5M2",     exact: true,  read: out(Float.self)),
]
if scaled == 1 {
    all.append(Spec(fn: "mm_f16xmxfp4_f32", a: aF16, b: bFP4, sa: nil, sb: sB, label: "FP16xMXFP4 (FP32)", exact: true, read: out(Float.self)))
    all.append(Spec(fn: "mm_mxfp4_f32",     a: aFP4, b: bFP4, sa: sA,  sb: sB, label: "MXFP4xMXFP4 (32)",  exact: true, read: out(Float.self)))
}

let want = opt["kernels"].map { Set($0.split(separator: ",").map(String.init)) }
var chosen: [Spec] = []
var psos: [MTLComputePipelineState] = []
for s in all {
    if let w = want, !w.contains(s.fn) { continue }
    guard let fn = lib.makeFunction(name: s.fn) else {
        if !csv { print("  [skip] \(s.fn) not in compiled metallib") }
        continue
    }
    do {
        psos.append(try device.makeComputePipelineState(function: fn))
        chosen.append(s)
    } catch {
        print("  [pso failed] \(s.fn): \(error)")
    }
}

let flops = 2.0 * Double(M) * Double(N) * Double(K)
var errs: [Double] = []
for (i, s) in chosen.enumerated() {
    _ = run(psos[i], s, iters: 1)
    errs.append(s.exact ? maxError(s) : .nan)
}

let warmStart = Date()
while Date().timeIntervalSince(warmStart) < warm {
    for (i, s) in chosen.enumerated() { _ = run(psos[i], s, iters: 3) }
}
let iters = max(3, Int(2e12 / flops))
var times = Array(repeating: [Double](), count: chosen.count)
for _ in 0..<reps {
    for (i, s) in chosen.enumerated() { times[i].append(run(psos[i], s, iters: iters)) }
}
func median(_ x: [Double]) -> Double { let s = x.sorted(); return s[s.count / 2] }
let fp16Idx = chosen.firstIndex { $0.fn == "mm_f16_f32" }
let i8Idx = chosen.firstIndex { $0.fn == "mm_i8_i32" }

for (i, s) in chosen.enumerated() {
    let med = median(times[i]), best = times[i].min()!
    let vs16 = fp16Idx.map { b in median(zip(times[b], times[i]).map { $0 / $1 }) } ?? .nan
    let vs8 = i8Idx.map { b in median(zip(times[b], times[i]).map { $0 / $1 }) } ?? .nan
    if csv {
        print([M, N, K, TM, TN, NSG, GROUP_M, BK].map(String.init).joined(separator: ",") +
              ",\(s.fn),\(med * 1e3),\(best * 1e3),\(flops / med / 1e12),\(vs16),\(vs8),\(errs[i])")
    } else {
        print(String(format: "  %-20@ median %8.3f ms  %6.1f T(FL)OPS  (best %6.1f)  vs FP16 %5.2fx  vs INT8 %5.2fx  max err %@",
                     s.label as NSString, med * 1e3, flops / med / 1e12, flops / best / 1e12, vs16, vs8,
                     errs[i].isNaN ? "n/a" : String(errs[i])))
    }
}
