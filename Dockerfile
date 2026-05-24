# syntax=docker/dockerfile:1
FROM debian:bookworm-slim AS build
ARG ZIG_VERSION=0.16.0
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl xz-utils ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL "https://ziglang.org/download/${ZIG_VERSION}/zig-x86_64-linux-${ZIG_VERSION}.tar.xz" \
    | tar -xJ -C /opt \
    && ln -s "/opt/zig-x86_64-linux-${ZIG_VERSION}/zig" /usr/local/bin/zig

WORKDIR /src
COPY build.zig build.zig.zon ./
COPY src ./src
COPY tools ./tools
RUN zig build --release=fast

# The binary is a static x86_64 musl build, so the runtime image only needs the
# binary plus the prebuilt model.
FROM debian:bookworm-slim AS runtime
WORKDIR /app
COPY --from=build /src/zig-out/bin/zordon /app/zordon
COPY --from=build /src/zig-out/bin/zordon-lb /app/zordon-lb
COPY data/model/ /app/model/

ENV PORT=8080
ENV ZORDON_MODEL_PATH=/app/model/references.i16.bin
ENV ZORDON_WORKERS=1

EXPOSE 8080
ENTRYPOINT ["/app/zordon"]
