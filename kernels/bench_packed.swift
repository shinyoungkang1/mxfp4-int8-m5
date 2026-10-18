// Compares SC-MXFP4 W4A8 with byte block shifts (w4a8.metal, 4.26 bits/weight) against 2-bit packed
// shifts (w4a8_packed.metal, 4.07 bits/weight): checks the unpacked INT8 weights are identical to the
// reference, then times both full pipelines (act-quant + unpack + INT8 GEMM + scaling) interleaved.
//
// Usage: bench_packed <testcase-dir from sim/make_testcase.py> [reps=15] [warm=1] [csv=1]
import Foundation
import Metal

let argv = Array(CommandLine.arguments.dropFirst())
guard let dir = argv.first else { print("usage: bench_packed <testcase-dir> [key=value ...]"); exit(2) }
var opt: [String: String] = [:]
for a in argv.dropFirst() {
    let kv = a.split(separator: "=", maxSplits: 1).map(String.init)
    if kv.count == 2 { opt[kv[0]] = kv[1] }
}
let reps = Int(opt["reps"] ?? "") ?? 15, warm = Double(opt["warm"] ?? "") ?? 1, csv = opt["csv"] == "1"
let meta = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: dir + "/meta.json"))) as! [String: Int]
let (M, N, K) = (meta["M"]!, meta["N"]!, meta["K"]!)
precondition(M % 128 == 0 && N % 64 == 0 && K % 128 == 0 && (N * K / 8) % 256 == 0 && (M * N / 4) % 256 == 0)

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let here = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
func library(_ file: String, _ macros: [String: Int] = [:]) -> MTLLibrary {
    let o = MTLCompileOptions()
    o.languageVersion = .version4_0
    o.preprocessorMacros = macros.mapValues { NSNumber(value: $0) }
    do { return try device.makeLibrary(source: String(contentsOf: here.appendingPathComponent(file), encoding: .utf8), options: o) }
    catch { print("compile \(file) failed:\n\(error)"); exit(1) }
}
let mxLib = library("w4a8.metal"), p2Lib = library("w4a8_packed.metal")
let mmLib = library("matmul.metal", ["TM": 128, "TN": 64, "NSG": 4, "GROUP_M": 8, "BK": 0])
func pso(_ lib: MTLLibrary, _ name: String) -> MTLComputePipelineState {
    try! device.makeComputePipelineState(function: lib.makeFunction(name: name)!)
}
func load(_ name: String) -> MTLBuffer {
    let d = try! Data(contentsOf: URL(fileURLWithPath: "\(dir)/\(name).bin"))
    return d.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: d.count, options: .storageModeShared)! }
}
let x = load("x"), codes = load("codes"), shifts = load("shifts"), shifts2 = load("shifts2"), wscale = load("wscale"), wq = load("wq")
let xq = device.makeBuffer(length: M * K, options: .storageModeShared)!
let xscale = device.makeBuffer(length: M * 4, options: .storageModeShared)!
let wByte = device.makeBuffer(length: N * K, options: .storageModeShared)!
let wP2 = device.makeBuffer(length: N * K, options: .storageModeShared)!
let acc = device.makeBuffer(length: M * N * 4, options: .storageModePrivate)!
let y = device.makeBuffer(length: M * N * 4, options: .storageModePrivate)!
var dims = SIMD3<UInt32>(UInt32(M), UInt32(N), UInt32(K))

struct Step { let pso: MTLComputePipelineState; let bufs: [MTLBuffer]; let grid: Int; let tg: Int; let grid2: Int }
let tg256 = 256
let actQuant = Step(pso: pso(mxLib, "act_quant"), bufs: [x, xq, xscale], grid: M, tg: 256, grid2: 1)
let unpackByte = Step(pso: pso(mxLib, "unpack_w_i8"), bufs: [codes, shifts, wByte], grid: N * K / 8 / tg256, tg: tg256, grid2: 1)
let unpackP2 = Step(pso: pso(p2Lib, "unpack_w_i8_p2"), bufs: [codes, shifts2, wP2], grid: N * K / 8 / tg256, tg: tg256, grid2: 1)
func gemm(_ w: MTLBuffer) -> Step { Step(pso: pso(mmLib, "mm_i8_i32"), bufs: [xq, w, acc], grid: N / 64, tg: 128, grid2: M / 128) }
let epilogue = Step(pso: pso(mxLib, "scale_epilogue"), bufs: [acc, xscale, wscale, y], grid: M * N / 4 / tg256, tg: tg256, grid2: 1)

func run(_ steps: [Step], iters: Int) -> Double {
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    for _ in 0..<iters {
        for s in steps {
            enc.setComputePipelineState(s.pso)
            for (i, b) in s.bufs.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
            enc.setBytes(&dims, length: MemoryLayout<SIMD3<UInt32>>.stride, index: s.bufs.count)
            enc.dispatchThreadgroups(MTLSize(width: s.grid, height: s.grid2, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: s.tg, height: 1, depth: 1))
        }
    }
    enc.endEncoding()
    cb.commit(); cb.waitUntilCompleted()
    if let e = cb.error { print("GPU error: \(e)"); exit(1) }
    return (cb.gpuEndTime - cb.gpuStartTime) / Double(iters)
}

// Correctness: both unpack kernels must reproduce the reference INT8 weights exactly.
_ = run([unpackByte, unpackP2], iters: 1)
let ref = wq.contents().assumingMemoryBound(to: Int8.self)
func mismatches(_ b: MTLBuffer) -> Int {
    let p = b.contents().assumingMemoryBound(to: Int8.self)
    return (0..<N * K).reduce(0) { $0 + (p[$1] == ref[$1] ? 0 : 1) }
}
let mByte = mismatches(wByte), mP2 = mismatches(wP2)
if !csv { print("  unpacked INT8 weights vs reference: byte shifts \(mByte) mismatches, 2-bit shifts \(mP2) mismatches (of \(N * K))") }
guard mByte == 0 && mP2 == 0 else { print("MISMATCH"); exit(1) }

let bitsByte = Double((N * K / 2 + N * K / 32 + N * 4) * 8) / Double(N * K)
let bitsP2 = Double((N * K / 2 + N * K / 128 + N * 4) * 8) / Double(N * K)
let pipelines: [(String, String, [Step])] = [
    ("w4a8_byte_shifts", String(format: "W4A8, byte shifts (%.2f bits/weight)", bitsByte), [actQuant, unpackByte, gemm(wByte), epilogue]),
    ("w4a8_2bit_shifts", String(format: "W4A8, 2-bit shifts (%.2f bits/weight)", bitsP2), [actQuant, unpackP2, gemm(wP2), epilogue]),
    ("unpack_byte", "  unpack only, byte shifts", [unpackByte]),
    ("unpack_2bit", "  unpack only, 2-bit shifts", [unpackP2]),
]
let t0 = Date()
while Date().timeIntervalSince(t0) < warm { for p in pipelines { _ = run(p.2, iters: 2) } }
let iters = max(3, Int(2e12 / (2.0 * Double(M) * Double(N) * Double(K))))
var times = Array(repeating: [Double](), count: pipelines.count)
for _ in 0..<reps { for (i, p) in pipelines.enumerated() { times[i].append(run(p.2, iters: iters)) } }
func median(_ v: [Double]) -> Double { let s = v.sorted(); return s[s.count / 2] }
for (i, p) in pipelines.enumerated() {
    let bits = p.0.contains("2bit") ? bitsP2 : bitsByte
    let ratio = median(zip(times[i - i % 2], times[i]).map { $0 / $1 })  // vs the byte-shift row of the pair
    if csv { print("\(M),\(N),\(K),\(p.0),\(bits),\(median(times[i]) * 1e3),\(ratio)") }
    else { print(String(format: "  %-44@ %8.3f ms   speed vs byte shifts %.3fx", p.1 as NSString, median(times[i]) * 1e3, ratio)) }
}
