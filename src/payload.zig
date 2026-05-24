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

/// Upper bound on `customer.known_merchants` entries. Real payloads carry a handful
/// (≤5 in the contest corpus); a request body never exceeds 8 KiB, so this cap is
/// generous. A payload that overflows it is rejected as malformed.
pub const max_known_merchants = 256;

pub const Error = error{Malformed};

/// Result of a successful parse. String fields (`id`, `mcc`, timestamps, …) are
/// slices that *borrow* from the input `body`; they are valid only while `body`
/// outlives the `Parsed`. `known_merchants` is the one heap allocation, owned here.
pub const Parsed = struct {
    value: Request,
    allocator: std.mem.Allocator,

    pub fn deinit(self: Parsed) void {
        self.allocator.free(self.value.customer.known_merchants);
    }
};

/// Parse a fraud-score request body. The schema is fixed and shallow, so this is a
/// single linear pass rather than a reflection-driven generic decode: no tokenizer,
/// no per-value allocation, no arena. Object members are accepted in any order.
///
/// Strings are returned as borrowed slices into `body` and are NOT unescaped — the
/// schema's strings (ids, MCC codes, ISO timestamps) never contain JSON escapes.
pub fn parse(allocator: std.mem.Allocator, body: []const u8) (Error || std.mem.Allocator.Error)!Parsed {
    var p = Parser{ .src = body };
    var merchants_buf: [max_known_merchants][]const u8 = undefined;
    var req = try p.parseRequest(&merchants_buf);

    // The merchant list lives in stack scratch; copy it onto the allocator so it
    // survives the call. Everything else borrows from `body`.
    req.customer.known_merchants = try allocator.dupe([]const u8, req.customer.known_merchants);
    return .{ .value = req, .allocator = allocator };
}

const Parser = struct {
    src: []const u8,
    i: usize = 0,

    fn skipWs(self: *Parser) void {
        while (self.i < self.src.len) : (self.i += 1) {
            switch (self.src[self.i]) {
                ' ', '\t', '\n', '\r' => {},
                else => return,
            }
        }
    }

    fn peek(self: *Parser) Error!u8 {
        if (self.i >= self.src.len) return error.Malformed;
        return self.src[self.i];
    }

    /// Skip whitespace, require `c` next, and step past it.
    fn expect(self: *Parser, c: u8) Error!void {
        self.skipWs();
        if (self.i >= self.src.len or self.src[self.i] != c) return error.Malformed;
        self.i += 1;
    }

    /// Consume the member separator: returns true if the object closed (`}`),
    /// false if another member follows (`,`).
    fn afterMember(self: *Parser) Error!bool {
        self.skipWs();
        const c = try self.peek();
        self.i += 1;
        return switch (c) {
            ',' => false,
            '}' => true,
            else => error.Malformed,
        };
    }

    /// Parse a JSON string, returning the raw inner bytes (no unescaping). Escapes
    /// are honored only enough to find the closing quote.
    fn string(self: *Parser) Error![]const u8 {
        self.skipWs();
        if (self.i >= self.src.len or self.src[self.i] != '"') return error.Malformed;
        self.i += 1;
        const start = self.i;
        while (self.i < self.src.len) : (self.i += 1) {
            switch (self.src[self.i]) {
                '\\' => self.i += 1, // skip the escaped byte; loop's += 1 skips the '\'
                '"' => {
                    const s = self.src[start..self.i];
                    self.i += 1;
                    return s;
                },
                else => {},
            }
        }
        return error.Malformed;
    }

    /// Span of a JSON number literal at the cursor.
    fn numberToken(self: *Parser) Error![]const u8 {
        self.skipWs();
        const start = self.i;
        while (self.i < self.src.len) : (self.i += 1) {
            switch (self.src[self.i]) {
                '0'...'9', '-', '+', '.', 'e', 'E' => {},
                else => break,
            }
        }
        if (self.i == start) return error.Malformed;
        return self.src[start..self.i];
    }

    fn float(self: *Parser) Error!f64 {
        return std.fmt.parseFloat(f64, try self.numberToken()) catch error.Malformed;
    }

    fn uint(self: *Parser) Error!u32 {
        return std.fmt.parseInt(u32, try self.numberToken(), 10) catch error.Malformed;
    }

    fn boolean(self: *Parser) Error!bool {
        self.skipWs();
        if (self.matchLit("true")) return true;
        if (self.matchLit("false")) return false;
        return error.Malformed;
    }

    fn matchLit(self: *Parser, lit: []const u8) bool {
        if (self.i + lit.len <= self.src.len and std.mem.eql(u8, self.src[self.i..][0..lit.len], lit)) {
            self.i += lit.len;
            return true;
        }
        return false;
    }

    fn stringArray(self: *Parser, buf: [][]const u8) Error![]const []const u8 {
        try self.expect('[');
        self.skipWs();
        if ((try self.peek()) == ']') {
            self.i += 1;
            return buf[0..0];
        }
        var n: usize = 0;
        while (true) {
            if (n >= buf.len) return error.Malformed;
            buf[n] = try self.string();
            n += 1;
            self.skipWs();
            const c = try self.peek();
            self.i += 1;
            switch (c) {
                ',' => continue,
                ']' => return buf[0..n],
                else => return error.Malformed,
            }
        }
    }

    /// Skip an arbitrary JSON value (used for unknown keys; the fixed schema has
    /// none, but this keeps the parser forgiving rather than brittle).
    fn skipValue(self: *Parser) Error!void {
        self.skipWs();
        switch (try self.peek()) {
            '"' => _ = try self.string(),
            '{', '[' => try self.skipContainer(),
            't' => if (!self.matchLit("true")) return error.Malformed,
            'f' => if (!self.matchLit("false")) return error.Malformed,
            'n' => if (!self.matchLit("null")) return error.Malformed,
            else => _ = try self.numberToken(),
        }
    }

    fn skipContainer(self: *Parser) Error!void {
        var depth: usize = 0;
        while (self.i < self.src.len) {
            switch (self.src[self.i]) {
                '"' => _ = try self.string(), // advances past the string
                '{', '[' => {
                    depth += 1;
                    self.i += 1;
                },
                '}', ']' => {
                    depth -= 1;
                    self.i += 1;
                    if (depth == 0) return;
                },
                else => self.i += 1,
            }
        }
        return error.Malformed;
    }

    fn parseRequest(self: *Parser, merchants_buf: [][]const u8) Error!Request {
        var r: Request = undefined;
        r.last_transaction = null; // optional: stays null if the key is absent
        var seen: u8 = 0;
        try self.expect('{');
        while (true) {
            const key = try self.string();
            try self.expect(':');
            if (std.mem.eql(u8, key, "id")) {
                r.id = try self.string();
                seen |= 1 << 0;
            } else if (std.mem.eql(u8, key, "transaction")) {
                r.transaction = try self.parseTransaction();
                seen |= 1 << 1;
            } else if (std.mem.eql(u8, key, "customer")) {
                r.customer = try self.parseCustomer(merchants_buf);
                seen |= 1 << 2;
            } else if (std.mem.eql(u8, key, "merchant")) {
                r.merchant = try self.parseMerchant();
                seen |= 1 << 3;
            } else if (std.mem.eql(u8, key, "terminal")) {
                r.terminal = try self.parseTerminal();
                seen |= 1 << 4;
            } else if (std.mem.eql(u8, key, "last_transaction")) {
                r.last_transaction = try self.parseLastTransaction();
            } else {
                try self.skipValue();
            }
            if (try self.afterMember()) break;
        }
        if (seen != 0b11111) return error.Malformed; // last_transaction excluded (optional)
        return r;
    }

    fn parseTransaction(self: *Parser) Error!Transaction {
        var t: Transaction = undefined;
        var seen: u8 = 0;
        try self.expect('{');
        while (true) {
            const key = try self.string();
            try self.expect(':');
            if (std.mem.eql(u8, key, "amount")) {
                t.amount = try self.float();
                seen |= 1 << 0;
            } else if (std.mem.eql(u8, key, "installments")) {
                t.installments = try self.uint();
                seen |= 1 << 1;
            } else if (std.mem.eql(u8, key, "requested_at")) {
                t.requested_at = try self.string();
                seen |= 1 << 2;
            } else {
                try self.skipValue();
            }
            if (try self.afterMember()) break;
        }
        if (seen != 0b111) return error.Malformed;
        return t;
    }

    fn parseCustomer(self: *Parser, merchants_buf: [][]const u8) Error!Customer {
        var c: Customer = undefined;
        var seen: u8 = 0;
        try self.expect('{');
        while (true) {
            const key = try self.string();
            try self.expect(':');
            if (std.mem.eql(u8, key, "avg_amount")) {
                c.avg_amount = try self.float();
                seen |= 1 << 0;
            } else if (std.mem.eql(u8, key, "tx_count_24h")) {
                c.tx_count_24h = try self.uint();
                seen |= 1 << 1;
            } else if (std.mem.eql(u8, key, "known_merchants")) {
                c.known_merchants = try self.stringArray(merchants_buf);
                seen |= 1 << 2;
            } else {
                try self.skipValue();
            }
            if (try self.afterMember()) break;
        }
        if (seen != 0b111) return error.Malformed;
        return c;
    }

    fn parseMerchant(self: *Parser) Error!Merchant {
        var m: Merchant = undefined;
        var seen: u8 = 0;
        try self.expect('{');
        while (true) {
            const key = try self.string();
            try self.expect(':');
            if (std.mem.eql(u8, key, "id")) {
                m.id = try self.string();
                seen |= 1 << 0;
            } else if (std.mem.eql(u8, key, "mcc")) {
                m.mcc = try self.string();
                seen |= 1 << 1;
            } else if (std.mem.eql(u8, key, "avg_amount")) {
                m.avg_amount = try self.float();
                seen |= 1 << 2;
            } else {
                try self.skipValue();
            }
            if (try self.afterMember()) break;
        }
        if (seen != 0b111) return error.Malformed;
        return m;
    }

    fn parseTerminal(self: *Parser) Error!Terminal {
        var t: Terminal = undefined;
        var seen: u8 = 0;
        try self.expect('{');
        while (true) {
            const key = try self.string();
            try self.expect(':');
            if (std.mem.eql(u8, key, "is_online")) {
                t.is_online = try self.boolean();
                seen |= 1 << 0;
            } else if (std.mem.eql(u8, key, "card_present")) {
                t.card_present = try self.boolean();
                seen |= 1 << 1;
            } else if (std.mem.eql(u8, key, "km_from_home")) {
                t.km_from_home = try self.float();
                seen |= 1 << 2;
            } else {
                try self.skipValue();
            }
            if (try self.afterMember()) break;
        }
        if (seen != 0b111) return error.Malformed;
        return t;
    }

    fn parseLastTransaction(self: *Parser) Error!?LastTransaction {
        self.skipWs();
        if (self.matchLit("null")) return null;

        var lt: LastTransaction = undefined;
        var seen: u8 = 0;
        try self.expect('{');
        while (true) {
            const key = try self.string();
            try self.expect(':');
            if (std.mem.eql(u8, key, "timestamp")) {
                lt.timestamp = try self.string();
                seen |= 1 << 0;
            } else if (std.mem.eql(u8, key, "km_from_current")) {
                lt.km_from_current = try self.float();
                seen |= 1 << 1;
            } else {
                try self.skipValue();
            }
            if (try self.afterMember()) break;
        }
        if (seen != 0b11) return error.Malformed;
        return lt;
    }
};

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
    try std.testing.expectEqual(@as(f64, 384.88), parsed.value.transaction.amount);
    try std.testing.expectEqualStrings("2026-03-11T20:23:35Z", parsed.value.transaction.requested_at);
    try std.testing.expectEqual(@as(f64, 769.76), parsed.value.customer.avg_amount);
    try std.testing.expectEqual(@as(usize, 2), parsed.value.customer.known_merchants.len);
    try std.testing.expectEqualStrings("MERC-009", parsed.value.customer.known_merchants[0]);
    try std.testing.expectEqualStrings("MERC-001", parsed.value.customer.known_merchants[1]);
    try std.testing.expectEqualStrings("5912", parsed.value.merchant.mcc);
    try std.testing.expect(!parsed.value.terminal.is_online);
    try std.testing.expect(parsed.value.terminal.card_present);
    try std.testing.expect(parsed.value.last_transaction != null);
    try std.testing.expectEqual(@as(f64, 18.8626479774), parsed.value.last_transaction.?.km_from_current);
}

test "null last_transaction and empty known_merchants" {
    const body =
        \\{
        \\  "id": "tx-1",
        \\  "transaction": {"amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z"},
        \\  "customer": {"avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": []},
        \\  "merchant": {"id": "MERC-016", "mcc": "5411", "avg_amount": 60.25},
        \\  "terminal": {"is_online": false, "card_present": true, "km_from_home": 29.2331036248},
        \\  "last_transaction": null
        \\}
    ;
    var parsed = try parse(std.testing.allocator, body);
    defer parsed.deinit();

    try std.testing.expect(parsed.value.last_transaction == null);
    try std.testing.expectEqual(@as(usize, 0), parsed.value.customer.known_merchants.len);
    try std.testing.expectEqual(@as(u32, 2), parsed.value.transaction.installments);
}

test "members in arbitrary order" {
    const body =
        \\{
        \\  "terminal": {"km_from_home": 1.5, "card_present": true, "is_online": true},
        \\  "last_transaction": {"km_from_current": 2.0, "timestamp": "2026-03-11T14:58:35Z"},
        \\  "merchant": {"avg_amount": 10.0, "mcc": "5411", "id": "MERC-001"},
        \\  "customer": {"known_merchants": ["A", "B"], "tx_count_24h": 9, "avg_amount": 5.0},
        \\  "transaction": {"requested_at": "2026-03-11T20:23:35Z", "installments": 4, "amount": 7.5},
        \\  "id": "reordered"
        \\}
    ;
    var parsed = try parse(std.testing.allocator, body);
    defer parsed.deinit();

    try std.testing.expectEqualStrings("reordered", parsed.value.id);
    try std.testing.expectEqual(@as(u32, 4), parsed.value.transaction.installments);
    try std.testing.expect(parsed.value.terminal.is_online);
    try std.testing.expectEqual(@as(f64, 2.0), parsed.value.last_transaction.?.km_from_current);
    try std.testing.expectEqual(@as(usize, 2), parsed.value.customer.known_merchants.len);
}

test "unknown keys are ignored" {
    const body =
        \\{
        \\  "extra": {"nested": [1, 2, {"a": "b"}], "s": "x}y"},
        \\  "id": "tx-x",
        \\  "transaction": {"amount": 1.0, "installments": 1, "requested_at": "2026-03-11T20:23:35Z", "note": "ignore"},
        \\  "customer": {"avg_amount": 1.0, "tx_count_24h": 1, "known_merchants": ["M"]},
        \\  "merchant": {"id": "M", "mcc": "5411", "avg_amount": 1.0},
        \\  "terminal": {"is_online": true, "card_present": false, "km_from_home": 1.0},
        \\  "trailing": [true, false, null]
        \\}
    ;
    var parsed = try parse(std.testing.allocator, body);
    defer parsed.deinit();

    try std.testing.expectEqualStrings("tx-x", parsed.value.id);
    try std.testing.expectEqualStrings("M", parsed.value.merchant.id);
}

test "rejects malformed and incomplete payloads" {
    const cases = [_][]const u8{
        "", // empty
        "{", // unterminated
        "not json",
        "{\"id\": \"x\"}", // missing required objects
        "{\"id\": \"x\", \"transaction\": {\"amount\": 1.0, \"installments\": 1}}", // missing requested_at
        "{\"id\": \"x\" \"transaction\": {}}", // missing comma
    };
    for (cases) |body| {
        try std.testing.expectError(error.Malformed, parse(std.testing.allocator, body));
    }
}
