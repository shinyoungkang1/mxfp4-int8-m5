// Linear-layer benchmark: SC-MXFP4 W4A8 vs W4A16 (same kernel, FP16 operands) vs FP16 and W8A8 GEMMs.
// Loads a test case written by sim/make_testcase.py, checks every kernel against the reference, then times
// all kernels interleaved (see bench.swift for the timing method).
//
// Usage: bench_layer <testcase-dir> [TM= TN= NSG= BK= GROUP_M= reps= warm= csv=1]
import Foundation
import Metal

let argv = Array(CommandLine.arguments.dropFirst())
guard let dir = argv.first else { print("usage: bench_layer <testcase-dir> [key=value ...]"); exit(2) }
var opt: [String: String] = [:]
for a in argv.dropFirst() {
    let kv = a.split(separator: "=", maxSplits: 1).map(String.init)
    guard kv.count == 2 else { print("bad argument \(a)"); exit(2) }
    opt[kv[0]] = kv[1]
}
func int(_ k: String, _ d: Int) -> Int { opt[k].flatMap { Int($0) } ?? d }
let (TM, TN, NSG, BK, GROUP_M) = (int("TM", 128), int("TN", 64), int("NSG", 4), int("BK", 128), int("GROUP_M", 8))
let reps = int("reps", 15), csv = int("csv", 0) == 1
let warm = Double(opt["warm"] ?? "1") ?? 1

let meta = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: dir + "/meta.json"))) as! [String: Int]
let (M, N, K) = (meta["M"]!, meta["N"]!, meta["K"]!)
guard M % TM == 0, N % TN == 0, K % BK == 0, BK % 32 == 0 else { print("shape/tile mismatch"); exit(2) }

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let here = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
func library(_ file: String, _ macros: [String: Int]) -> MTLLibrary {
    let o = MTLCompileOptions()
    o.languageVersion = .version4_0
    o.preprocessorMacros = macros.mapValues { NSNumber(value: $0) }
    do { return try device.makeLibrary(source: String(contentsOf: here.appendingPathComponent(file), encoding: .utf8), options: o) }
    catch { print("compile \(file) failed:\n\(error)"); exit(1) }
}
let mxLib = library("w4a8.metal", ["TM": TM, "TN": TN, "NSG": NSG, "BK": BK, "GROUP_M": GROUP_M])
let mmLib = library("matmul.metal", ["TM": 128, "TN": 64, "NSG": 4, "GROUP_M": 8, "BK": 0])
func pso(_ lib: MTLLibrary, _ name: String) -> MTLComputePipelineState {
    try! device.makeComputePipelineState(function: lib.makeFunction(name: name)!)
}

func load(_ name: String) -> MTLBuffer {
    let d = try! Data(contentsOf: URL(fileURLWithPath: "\(dir)/\(name).bin"))
    return d.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: d.count, options: .storageModeShared)! }
}
func arr<T>(_ b: MTLBuffer, _: T.Type) -> UnsafeMutableBufferPointer<T> {
    UnsafeMutableBufferPointer(start: b.contents().assumingMemoryBound(to: T.self), count: b.length / MemoryLayout<T>.stride)
}
let x = load("x"), codes = load("codes"), shifts = load("shifts"), wscale = load("wscale")
let xqRef = load("xq"), xscaleRef = load("xscale"), wq = load("wq"), wdq = load("wdq"), yRef = load("y_ref")
let xq = device.makeBuffer(length: M * K, options: .storageModeShared)!
let xscale = device.makeBuffer(length: M * 4, options: .storageModeShared)!
let y = device.makeBuffer(length: M * N * 4, options: .storageModeShared)!
var dims = SIMD3<UInt32>(UInt32(M), UInt32(N), UInt32(K))

struct Step { let pso: MTLComputePipelineState; let bufs: [MTLBuffer?]; let grid: MTLSize; let tg: MTLSize }
let gemmGrid = MTLSize(width: N / TN, height: M / TM, depth: 1)
let actQuant = Step(pso: pso(mxLib, "act_quant"), bufs: [x, xq, xscale], grid: MTLSize(width: M, height: 1, depth: 1),
                    tg: MTLSize(width: 256, height: 1, depth: 1))
func mxStep(_ name: String, _ a: MTLBuffer, _ xs: MTLBuffer?) -> Step {
    Step(pso: pso(mxLib, name), bufs: [a, codes, shifts, wscale, xs, y], grid: gemmGrid, tg: MTLSize(width: 32 * NSG, height: 1, depth: 1))
}
func mmStep(_ name: String, _ a: MTLBuffer, _ b: MTLBuffer) -> Step {
    Step(pso: pso(mmLib, name), bufs: [a, b, y], grid: MTLSize(width: N / 64, height: M / 128, depth: 1),
         tg: MTLSize(width: 128, height: 1, depth: 1))
}

func encode(_ enc: MTLComputeCommandEncoder, _ s: Step) {
    enc.setComputePipelineState(s.pso)
    for (i, b) in s.bufs.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
    enc.setBytes(&dims, length: MemoryLayout<SIMD3<UInt32>>.stride, index: s.bufs.count)
    enc.dispatchThreadgroups(s.grid, threadsPerThreadgroup: s.tg)
}
/// Runs a pipeline (one or more steps, e.g. activation quantization then GEMM) `iters` times; returns s/iter.
func run(_ steps: [Step], iters: Int) -> Double {
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    for _ in 0..<iters { for s in steps { encode(enc, s) } }
    enc.endEncoding()
    cb.commit(); cb.waitUntilCompleted()
    if let e = cb.error { print("GPU error: \(e)"); exit(1) }
    return (cb.gpuEndTime - cb.gpuStartTime) / Double(iters)
}

// ---- correctness -------------------------------------------------------------------------------
var report: [String] = []
_ = run([actQuant], iters: 1)
do {
    let a = arr(xq, Int8.self), r = arr(xqRef, Int8.self)
    var mism = 0, maxd = 0
    for i in 0..<M * K where a[i] != r[i] { mism += 1; maxd = max(maxd, abs(Int(a[i]) - Int(r[i]))) }
    let s = arr(xscale, Float.self), sr = arr(xscaleRef, Float.self)
    let sErr = (0..<M).map { abs(s[$0] - sr[$0]) / sr[$0] }.max()!
    report.append(String(format: "act_quant: %d/%d codes differ from float64 reference (max |diff| %d), scale rel err %.1e", mism, M * K, maxd, sErr))
}
// GEMMs on the reference INT8 activations, so each GEMM is checked in isolation.
func checkY(_ label: String, sampledRef: ((Int, Int) -> Double)?) {
    let out = arr(y, Float.self), ref = arr(yRef, Float.self)
    var maxRel = 0.0, scale = 0.0
    if let f = sampledRef {
        for _ in 0..<256 {
            let i = Int.random(in: 0..<M), j = Int.random(in: 0..<N)
            let r = f(i, j)
            scale = max(scale, abs(r)); maxRel = max(maxRel, abs(Double(out[i * N + j]) - r))
        }
    } else {
        for i in 0..<M * N { scale = max(scale, abs(Double(ref[i]))); maxRel = max(maxRel, abs(Double(out[i] - ref[i]))) }
    }
    report.append(String(format: "%@: max abs err / max |y| = %.2e", label as NSString, maxRel / scale))
}
let w4a8Ref = mxStep("mx_w4a8", xqRef, xscaleRef)
_ = run([w4a8Ref], iters: 1); checkY("mx_w4a8 vs exact reference (all outputs)", sampledRef: nil)
let xh = arr(x, Float16.self), wdh = arr(wdq, Float16.self), wsc = arr(wscale, Float.self), wqa = arr(wq, Int8.self)
let fp16Ref: (Int, Int) -> Double = { i, j in (0..<K).reduce(0.0) { $0 + Double(xh[i * K + $1]) * Double(wdh[j * K + $1]) } }
let w4a16 = mxStep("mx_w4a16", x, nil)
_ = run([w4a16], iters: 1)
checkY("mx_w4a16 vs float64 x @ Wdq^T (256 samples)", sampledRef: { i, j in
    Double(wsc[j]) * (0..<K).reduce(0.0) { $0 + Double(xh[i * K + $1]) * Double(wqa[j * K + $1]) } })
let fp16 = mmStep("mm_f16_f32", x, wdq)
_ = run([fp16], iters: 1); checkY("FP16 GEMM vs float64 (256 samples)", sampledRef: fp16Ref)
let w8a8 = mmStep("mm_i8_i32", xq, wq)  // int32 out; checked in bench.swift, timed only here

// Two-pass path: unpack the packed weights once into a scratch buffer, then GEMM from device memory.
let wI8 = device.makeBuffer(length: N * K, options: .storageModePrivate)!
let wF16 = device.makeBuffer(length: N * K * 2, options: .storageModePrivate)!
precondition((N * K / 8) % 256 == 0)
let unpackGrid = MTLSize(width: N * K / 8 / 256, height: 1, depth: 1), unpackTG = MTLSize(width: 256, height: 1, depth: 1)  // threadgroups
let unpackI8 = Step(pso: pso(mxLib, "unpack_w_i8"), bufs: [codes, shifts, wI8], grid: unpackGrid, tg: unpackTG)
let unpackF16 = Step(pso: pso(mxLib, "unpack_w_f16"), bufs: [codes, shifts, wF16], grid: unpackGrid, tg: unpackTG)
func scaledGemm(_ name: String, _ a: MTLBuffer, _ w: MTLBuffer, _ xs: MTLBuffer?) -> Step {
    Step(pso: pso(mxLib, name), bufs: [a, w, wscale, xs, y], grid: gemmGrid, tg: MTLSize(width: 32 * NSG, height: 1, depth: 1))
}
let gemmI8 = scaledGemm("gemm_i8_scaled", xq, wI8, xscale), gemmF16 = scaledGemm("gemm_f16_scaled", x, wF16, nil)
// Variant: plain INT32-output GEMM (matmul.metal) + separate scaling pass.
let accI32 = device.makeBuffer(length: M * N * 4, options: .storageModePrivate)!
let gemmI8Plain = Step(pso: pso(mmLib, "mm_i8_i32"), bufs: [xq, wI8, accI32], grid: MTLSize(width: N / 64, height: M / 128, depth: 1),
                       tg: MTLSize(width: 128, height: 1, depth: 1))
precondition((M * N / 4) % 256 == 0)
let epilogue = Step(pso: pso(mxLib, "scale_epilogue"), bufs: [accI32, xscale, wscale, y],
                    grid: MTLSize(width: M * N / 4 / 256, height: 1, depth: 1), tg: MTLSize(width: 256, height: 1, depth: 1))
func dispatchEach(_ steps: [Step]) { let cb = queue.makeCommandBuffer()!; let e = cb.makeComputeCommandEncoder()!
    steps.forEach { encode(e, $0) }; e.endEncoding(); cb.commit(); cb.waitUntilCompleted() }
dispatchEach([actQuant])  // restore xq from the activation kernel (equal to the reference up to rare ±1 ties)
dispatchEach([unpackI8, scaledGemm("gemm_i8_scaled", xqRef, wI8, xscaleRef)])
checkY("2-pass W4A8 (unpack + INT8 GEMM) vs exact reference (all outputs)", sampledRef: nil)
dispatchEach([unpackI8,
              Step(pso: gemmI8Plain.pso, bufs: [xqRef, wI8, accI32], grid: gemmI8Plain.grid, tg: gemmI8Plain.tg),
              Step(pso: epilogue.pso, bufs: [accI32, xscaleRef, wscale, y], grid: epilogue.grid, tg: epilogue.tg)])
checkY("2-pass W4A8, plain GEMM + scaling pass vs exact reference (all outputs)", sampledRef: nil)
dispatchEach([actQuant])
dispatchEach([unpackF16, gemmF16])
checkY("2-pass W4A16 (unpack + FP16 GEMM) vs float64 (256 samples)", sampledRef: { i, j in
    Double(wsc[j]) * (0..<K).reduce(0.0) { $0 + Double(xh[i * K + $1]) * Double(wqa[j * K + $1]) } })
if !csv { report.forEach { print("  " + $0) } }

// ---- timing ------------------------------------------------------------------------------------
let pipelines: [(String, String, [Step])] = [
    ("fp16", "FP16 weights, FP16 GEMM (16-bit weights)", [fp16]),
    ("w4a16", "SC-MXFP4 W4A16: unpack to FP16 tiles", [w4a16]),
    ("w4a8", "SC-MXFP4 W4A8: act-quant + unpack to INT8 tiles", [actQuant, mxStep("mx_w4a8", xq, xscale)]),
    ("w4a8_gemm", "  W4A8 GEMM only", [mxStep("mx_w4a8", xq, xscale)]),
    ("act_quant", "  activation quantization only", [actQuant]),
    ("w8a8", "W8A8 ceiling: act-quant + INT8 GEMM (8-bit weights)", [actQuant, w8a8]),
    ("w4a16_2pass", "2-pass W4A16: unpack to FP16 scratch + FP16 GEMM", [unpackF16, gemmF16]),
    ("w4a8_2pass", "2-pass W4A8: act-quant + unpack to INT8 scratch + INT8 GEMM", [actQuant, unpackI8, gemmI8]),
    ("w4a8_2pass_sep", "2-pass W4A8, plain INT8 GEMM + separate scaling pass", [actQuant, unpackI8, gemmI8Plain, epilogue]),
    ("unpack_i8", "  INT8 unpack only", [unpackI8]),
    ("epilogue", "  scaling pass only", [epilogue]),
]
let flops = 2.0 * Double(M) * Double(N) * Double(K)
let iters = max(3, Int(2e12 / flops))
let t0 = Date()
while Date().timeIntervalSince(t0) < warm { for p in pipelines { _ = run(p.2, iters: 2) } }
var times = Array(repeating: [Double](), count: pipelines.count)
for _ in 0..<reps { for (i, p) in pipelines.enumerated() { times[i].append(run(p.2, iters: iters)) } }
func median(_ v: [Double]) -> Double { let s = v.sorted(); return s[s.count / 2] }
for (i, p) in pipelines.enumerated() {
    let med = median(times[i])
    let vsFp16 = median(zip(times[0], times[i]).map { $0 / $1 })
    let vsW4a16 = median(zip(times[1], times[i]).map { $0 / $1 })
    if csv {
        print("\(M),\(N),\(K),\(TM),\(TN),\(NSG),\(BK),\(GROUP_M),\(p.0),\(med * 1e3),\(flops / med / 1e12),\(vsFp16),\(vsW4a16)")
    } else {
        print(String(format: "  %-52@ %8.3f ms  %6.1f TFLOPS-eq  vs FP16 %5.2fx  vs W4A16 %5.2fx",
                     p.1 as NSString, med * 1e3, flops / med / 1e12, vsFp16, vsW4a16))
    }
}
