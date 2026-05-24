# syntax=docker/dockerfile:1
FROM debian:bookworm-slim AS build
ARG ZIG_VERSION=0.16.0
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl xz-utils ca-certificates gzip \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL "https://ziglang.org/download/${ZIG_VERSION}/zig-x86_64-linux-${ZIG_VERSION}.tar.xz" \
    | tar -xJ -C /opt \
    && ln -s "/opt/zig-x86_64-linux-${ZIG_VERSION}/zig" /usr/local/bin/zig

WORKDIR /src
COPY build.zig build.zig.zon ./
COPY src ./src
COPY tools ./tools
RUN zig build --release=fast

# Generate the model from the committed compressed reference set (decompress +
# preprocess) so the build is self-contained — no prebuilt model in the context.
# The 298 MB intermediate JSON is removed in the same layer.
COPY data/resources/references.json.gz ./data/resources/references.json.gz
RUN mkdir -p data/model \
    && gzip -dc data/resources/references.json.gz > references.json \
    && zig build --release=fast preprocess -- references.json data/model/references.i16.bin \
    && rm references.json

# Static x86_64 musl binaries, so the runtime image only needs the two binaries
# plus the generated model.
FROM debian:bookworm-slim AS runtime
WORKDIR /app
COPY --from=build /src/zig-out/bin/zordon /app/zordon
COPY --from=build /src/zig-out/bin/zordon-lb /app/zordon-lb
COPY --from=build /src/data/model/references.i16.bin /app/model/references.i16.bin

ENV PORT=8080
ENV ZORDON_MODEL_PATH=/app/model/references.i16.bin
ENV ZORDON_WORKERS=1

EXPOSE 8080
ENTRYPOINT ["/app/zordon"]
