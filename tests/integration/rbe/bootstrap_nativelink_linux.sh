#!/usr/bin/env bash
# Download NativeLink (linux musl) and start a single-node RE endpoint on :1985.
# Intended to run inside any Linux (qemu guest, CI VM, or local Linux).
set -euo pipefail

PORT="${RBE_PORT:-1985}"
CACHE="${NATIVELINK_CACHE:-$HOME/.cache/rules_cuda-rbe}"
VER="${NATIVELINK_VERSION:-1.6.4}"
mkdir -p "$CACHE/content" "$CACHE/index" "$CACHE/bin"
cd "$CACHE"

BIN="$CACHE/bin/nativelink"
if [[ ! -x "$BIN" ]]; then
  TGZ="nativelink-${VER}-x86_64-unknown-linux-musl.tar.gz"
  URL="https://github.com/TraceMachina/nativelink/releases/download/v${VER}/${TGZ}"
  echo "Downloading $URL"
  curl -fL --retry 5 -o "$TGZ" "$URL"
  tar -xzf "$TGZ" -C "$CACHE/bin"
  # tarball layout may nest
  if [[ ! -x "$BIN" ]]; then
    found=$(find "$CACHE/bin" -type f -name nativelink | head -n 1)
    if [[ -n "$found" ]]; then
      ln -sfn "$found" "$BIN"
    fi
  fi
  chmod +x "$BIN" || true
fi
"$BIN" --version 2>/dev/null || "$BIN" -V 2>/dev/null || ls -la "$CACHE/bin"

# Minimal JSON — try common nativelink 0.6+/1.x basic_cas style if present in package.
# Fall back to generating a tiny config from --help examples.
CONFIG="$CACHE/basic_cas.json"
if [[ ! -f "$CONFIG" ]]; then
  # NativeLink 1.x often ships examples; search
  ex=$(find "$CACHE" -name '*basic*.json' 2>/dev/null | head -n 1 || true)
  if [[ -n "$ex" ]]; then
    cp "$ex" "$CONFIG"
  else
    cat >"$CONFIG" <<EOF
{
  "stores": {
    "AC_MAIN_STORE": {
      "filesystem": {
        "content_path": "${CACHE}/content",
        "eviction": { "max_bytes": 10000000000 }
      }
    },
    "CAS_MAIN_STORE": {
      "filesystem": {
        "content_path": "${CACHE}/content",
        "eviction": { "max_bytes": 10000000000 }
      }
    }
  },
  "schedulers": {
    "MAIN_SCHEDULER": {
      "simple": {
        "supported_platform_properties": {
          "properties": {
            "OSFamily": { "values": ["Linux", ""] },
            "container-image": { "values": ["", "*"] }
          }
        }
      }
    }
  },
  "workers": {
    "WORKER": {
      "local": {
        "platform_properties": {
          "OSFamily": "Linux"
        },
        "entry": {
          "cas_store": "CAS_MAIN_STORE",
          "scheduler": "MAIN_SCHEDULER"
        }
      }
    }
  },
  "servers": [{
    "listener": {
      "http": { "socket_address": "0.0.0.0:${PORT}" }
    },
    "services": {
      "cas": "CAS_MAIN_STORE",
      "ac": "AC_MAIN_STORE",
      "execution": "MAIN_SCHEDULER"
    }
  }]
}
EOF
  fi
fi

echo "Starting nativelink on 0.0.0.0:${PORT} config=${CONFIG}"
exec "$BIN" "$CONFIG"
