const std = @import("std");
const payload = @import("payload.zig");
const time = @import("time.zig");

pub const dims = 14;
/// Vectors are stored padded to 16 lanes so distance math maps onto a single
/// `@Vector(16, i16)` (AVX2-friendly). Lanes 14 and 15 are always 0 in both the
/// query and every reference, so they contribute 0 to the squared distance.
pub const stored_dims = 16;
pub const scale: i32 = 10_000;
pub const Vector = [dims]f32;
pub const QuantizedVector = [stored_dims]i16;

const max_amount = 10_000.0;
const max_installments = 12.0;
const amount_vs_avg_ratio = 10.0;
const max_minutes = 1_440.0;
const max_km = 1_000.0;
const max_tx_count_24h = 20.0;
const max_merchant_avg_amount = 10_000.0;

pub fn fromRequest(req: payload.Request) !Vector {
    const requested_at = try time.parseIsoUtc(req.transaction.requested_at);

    var out: Vector = undefined;
    out[0] = clamp01(req.transaction.amount / max_amount);
    out[1] = clamp01(@as(f64, @floatFromInt(req.transaction.installments)) / max_installments);
    out[2] = if (req.customer.avg_amount <= 0.0)
        1.0
    else
        clamp01((req.transaction.amount / req.customer.avg_amount) / amount_vs_avg_ratio);
    out[3] = @as(f32, @floatCast(@as(f64, @floatFromInt(requested_at.hour)) / 23.0));
    out[4] = @as(f32, @floatCast(@as(f64, @floatFromInt(time.dayOfWeekMondayZero(requested_at))) / 6.0));

    if (req.last_transaction) |last| {
        const last_ts = try time.parseIsoUtc(last.timestamp);
        out[5] = clamp01(@as(f64, @floatFromInt(time.minutesBetween(requested_at, last_ts))) / max_minutes);
        out[6] = clamp01(last.km_from_current / max_km);
    } else {
        out[5] = -1.0;
        out[6] = -1.0;
    }

    out[7] = clamp01(req.terminal.km_from_home / max_km);
    out[8] = clamp01(@as(f64, @floatFromInt(req.customer.tx_count_24h)) / max_tx_count_24h);
    out[9] = if (req.terminal.is_online) 1.0 else 0.0;
    out[10] = if (req.terminal.card_present) 1.0 else 0.0;
    out[11] = if (knownMerchant(req.customer.known_merchants, req.merchant.id)) 0.0 else 1.0;
    out[12] = @as(f32, @floatCast(mccRisk(req.merchant.mcc)));
    out[13] = clamp01(req.merchant.avg_amount / max_merchant_avg_amount);
    return out;
}

pub fn quantize(v: Vector) QuantizedVector {
    var out: QuantizedVector = @splat(0);
    for (out[0..dims], v) |*slot, value| {
        const bounded = std.math.clamp(value, -1.0, 1.0);
        slot.* = @intFromFloat(@round(bounded * @as(f32, @floatFromInt(scale))));
    }
    return out;
}

/// The 4-bit stage-1 bucket key: is_online | card_present | unknown_merchant |
/// has_history. These dimensions are quantized to exactly 0 or `scale` (and `-scale`
/// for the no-history sentinel), so the comparisons below are exact.
pub fn bucketKey(q: QuantizedVector) u4 {
    const online: u4 = @intFromBool(q[9] > scale >> 1);
    const card_present: u4 = @intFromBool(q[10] > scale >> 1);
    const unknown: u4 = @intFromBool(q[11] > scale >> 1);
    const has_history: u4 = @intFromBool(q[5] > -(scale >> 1));
    return (online << 3) | (card_present << 2) | (unknown << 1) | has_history;
}

pub fn quantizeValue(value: f64) i16 {
    const bounded = std.math.clamp(value, -1.0, 1.0);
    return @intFromFloat(@round(bounded * @as(f64, @floatFromInt(scale))));
}

pub fn mccRisk(mcc: []const u8) f64 {
    if (std.mem.eql(u8, mcc, "5411")) return 0.15;
    if (std.mem.eql(u8, mcc, "5812")) return 0.30;
    if (std.mem.eql(u8, mcc, "5912")) return 0.20;
    if (std.mem.eql(u8, mcc, "5944")) return 0.45;
    if (std.mem.eql(u8, mcc, "7801")) return 0.80;
    if (std.mem.eql(u8, mcc, "7802")) return 0.75;
    if (std.mem.eql(u8, mcc, "7995")) return 0.85;
    if (std.mem.eql(u8, mcc, "4511")) return 0.35;
    if (std.mem.eql(u8, mcc, "5311")) return 0.25;
    if (std.mem.eql(u8, mcc, "5999")) return 0.50;
    return 0.50;
}

fn knownMerchant(known: []const []const u8, merchant_id: []const u8) bool {
    for (known) |id| {
        if (std.mem.eql(u8, id, merchant_id)) return true;
    }
    return false;
}

fn clamp01(value: f64) f32 {
    return @as(f32, @floatCast(std.math.clamp(value, 0.0, 1.0)));
}

test "vectorizes documented legitimate example" {
    const body =
        \\{
        \\  "id": "tx-1329056812",
        \\  "transaction": {"amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z"},
        \\  "customer": {"avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"]},
        \\  "merchant": {"id": "MERC-016", "mcc": "5411", "avg_amount": 60.25},
        \\  "terminal": {"is_online": false, "card_present": true, "km_from_home": 29.2331036248},
        \\  "last_transaction": null
        \\}
    ;
    var parsed = try payload.parse(std.testing.allocator, body);
    defer parsed.deinit();

    const v = try fromRequest(parsed.value);
    try expectApprox(0.0041, v[0]);
    try expectApprox(0.1667, v[1]);
    try expectApprox(0.05, v[2]);
    try expectApprox(0.7826, v[3]);
    try expectApprox(0.3333, v[4]);
    try expectApprox(-1.0, v[5]);
    try expectApprox(-1.0, v[6]);
    try expectApprox(0.0292, v[7]);
    try expectApprox(0.15, v[8]);
    try expectApprox(0.0, v[9]);
    try expectApprox(1.0, v[10]);
    try expectApprox(0.0, v[11]);
    try expectApprox(0.15, v[12]);
    try expectApprox(0.006, v[13]);
}

test "mcc default risk" {
    try std.testing.expectEqual(@as(f64, 0.50), mccRisk("0000"));
}

fn expectApprox(expected: f32, actual: f32) !void {
    try std.testing.expectApproxEqAbs(expected, actual, 0.0001);
}
