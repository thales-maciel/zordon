FROM ghcr.io/ziglang/zig:master AS build

WORKDIR /src
COPY build.zig build.zig.zon ./
COPY src ./src
COPY tools ./tools
RUN zig build --release=fast

FROM debian:bookworm-slim AS runtime

WORKDIR /app
COPY --from=build /src/zig-out/bin/zordon /app/zordon
COPY data/model/ /app/model/

ENV PORT=8080
ENV ZORDON_MODEL_PATH=/app/model/references.i16.bin

EXPOSE 8080
ENTRYPOINT ["/app/zordon"]
