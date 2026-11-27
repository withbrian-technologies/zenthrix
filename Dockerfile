# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    make \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY CMakeLists.txt pyproject.toml README.md LICENSE ./
COPY core ./core
COPY kernels ./kernels
COPY python ./python

RUN cmake -S . -B build/native \
      -DCMAKE_BUILD_TYPE=Release \
      -DZENTHRIX_BUILD_TESTS=OFF \
  && cmake --build build/native --parallel "$(nproc)"

RUN python -m venv /opt/venv \
  && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
  && /opt/venv/bin/pip install --no-cache-dir .


FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    ZENTHRIX_NATIVE_LIB="/usr/local/lib/libzenthrix.so" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /src/build/native/libzenthrix.so /usr/local/lib/libzenthrix.so

RUN ldconfig \
  && useradd --create-home --shell /bin/bash zenthrix

USER zenthrix
WORKDIR /workspace

ENTRYPOINT ["zenthrix"]
CMD ["--help"]
