const std = @import("std");

pub const Request = struct {
    id: []const u8,
    transaction: Transaction,
    customer: Customer,
    merchant: Merchant,
    terminal: Terminal,
    last_transaction: ?LastTransaction,
};

pub const Transaction = struct {
    amount: f64,
    installments: u32,
    requested_at: []const u8,
};

pub const Customer = struct {
    avg_amount: f64,
    tx_count_24h: u32,
    known_merchants: []const []const u8,
};

pub const Merchant = struct {
    id: []const u8,
    mcc: []const u8,
    avg_amount: f64,
};

pub const Terminal = struct {
    is_online: bool,
    card_present: bool,
    km_from_home: f64,
};

pub const LastTransaction = struct {
    timestamp: []const u8,
    km_from_current: f64,
};

pub fn parse(allocator: std.mem.Allocator, body: []const u8) !std.json.Parsed(Request) {
    return std.json.parseFromSlice(Request, allocator, body, .{});
}

test "parse sample payload" {
    const body =
        \\{
        \\  "id": "tx-smoke-001",
        \\  "transaction": {
        \\    "amount": 384.88,
        \\    "installments": 3,
        \\    "requested_at": "2026-03-11T20:23:35Z"
        \\  },
        \\  "customer": {
        \\    "avg_amount": 769.76,
        \\    "tx_count_24h": 3,
        \\    "known_merchants": ["MERC-009", "MERC-001"]
        \\  },
        \\  "merchant": {
        \\    "id": "MERC-001",
        \\    "mcc": "5912",
        \\    "avg_amount": 298.95
        \\  },
        \\  "terminal": {
        \\    "is_online": false,
        \\    "card_present": true,
        \\    "km_from_home": 13.7090520965
        \\  },
        \\  "last_transaction": {
        \\    "timestamp": "2026-03-11T14:58:35Z",
        \\    "km_from_current": 18.8626479774
        \\  }
        \\}
    ;

    var parsed = try parse(std.testing.allocator, body);
    defer parsed.deinit();

    try std.testing.expectEqualStrings("tx-smoke-001", parsed.value.id);
    try std.testing.expectEqual(@as(u32, 3), parsed.value.transaction.installments);
    try std.testing.expect(parsed.value.last_transaction != null);
}
