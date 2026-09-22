# Software twin of a Phase 8 node (plan: docs/plans/phase8-software-twin.md).
#
# One image for every role -- aggregator, Pi client/runtime, cloud receiver, virtual sensor --
# selected by the compose `command`. Built for linux/arm64 on an Apple-silicon host so the
# stack is exercised on the Raspberry Pi's architecture (aarch64 CPU wheels of torch); on an
# x86 host pass --platform linux/amd64 and accept that the architecture check is lost.
# Timings from a container are NEVER Pi timings; every manifest is tagged with --platform.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends openssl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests/kat ./tests/kat
# CPU-only torch keeps the image small; the aarch64 wheel is what a Pi would install too.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -e .

ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
ENTRYPOINT ["asa"]
