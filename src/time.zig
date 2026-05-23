const std = @import("std");

pub const Timestamp = struct {
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: u8,
};

pub const Error = error{InvalidTimestamp};

pub fn parseIsoUtc(s: []const u8) Error!Timestamp {
    if (s.len != 20) return error.InvalidTimestamp;
    if (s[4] != '-' or s[7] != '-' or s[10] != 'T' or s[13] != ':' or s[16] != ':' or s[19] != 'Z') {
        return error.InvalidTimestamp;
    }

    const year = parseDigits(i32, s[0..4]) catch return error.InvalidTimestamp;
    const month = parseDigits(u8, s[5..7]) catch return error.InvalidTimestamp;
    const day = parseDigits(u8, s[8..10]) catch return error.InvalidTimestamp;
    const hour = parseDigits(u8, s[11..13]) catch return error.InvalidTimestamp;
    const minute = parseDigits(u8, s[14..16]) catch return error.InvalidTimestamp;
    const second = parseDigits(u8, s[17..19]) catch return error.InvalidTimestamp;

    if (month < 1 or month > 12) return error.InvalidTimestamp;
    if (day < 1 or day > 31) return error.InvalidTimestamp;
    if (hour > 23 or minute > 59 or second > 60) return error.InvalidTimestamp;

    return .{
        .year = year,
        .month = month,
        .day = day,
        .hour = hour,
        .minute = minute,
        .second = second,
    };
}

pub fn epochSeconds(ts: Timestamp) i64 {
    return daysFromCivil(ts.year, ts.month, ts.day) * 86_400 +
        @as(i64, ts.hour) * 3_600 +
        @as(i64, ts.minute) * 60 +
        @as(i64, ts.second);
}

pub fn minutesBetween(later: Timestamp, earlier: Timestamp) i64 {
    return @divTrunc(epochSeconds(later) - epochSeconds(earlier), 60);
}

pub fn dayOfWeekMondayZero(ts: Timestamp) u8 {
    const days = daysFromCivil(ts.year, ts.month, ts.day);
    return @intCast(@mod(days + 3, 7));
}

fn parseDigits(comptime T: type, s: []const u8) !T {
    var value: T = 0;
    for (s) |ch| {
        if (ch < '0' or ch > '9') return error.InvalidCharacter;
        value = value * 10 + @as(T, @intCast(ch - '0'));
    }
    return value;
}

fn daysFromCivil(year: i32, month: u8, day: u8) i64 {
    var y = year;
    const m_i32: i32 = month;
    y -= if (m_i32 <= 2) 1 else 0;
    const era = @divFloor(y, 400);
    const yoe = y - era * 400;
    const mp = m_i32 + if (m_i32 > 2) @as(i32, -3) else @as(i32, 9);
    const doy = @divFloor(153 * mp + 2, 5) + @as(i32, day) - 1;
    const doe = yoe * 365 + @divFloor(yoe, 4) - @divFloor(yoe, 100) + doy;
    return @as(i64, era) * 146_097 + doe - 719_468;
}

test "day of week follows contest convention" {
    const wednesday = try parseIsoUtc("2026-03-11T20:23:35Z");
    try std.testing.expectEqual(@as(u8, 2), dayOfWeekMondayZero(wednesday));

    const saturday = try parseIsoUtc("2026-03-14T05:15:12Z");
    try std.testing.expectEqual(@as(u8, 5), dayOfWeekMondayZero(saturday));
}

test "minutes between timestamps" {
    const later = try parseIsoUtc("2026-03-11T20:23:35Z");
    const earlier = try parseIsoUtc("2026-03-11T14:58:35Z");
    try std.testing.expectEqual(@as(i64, 325), minutesBetween(later, earlier));
}
