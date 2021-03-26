export SMOL_THREADS=64

cargo run --bin kajiya --release -- --scene $* --width 1920 --height 1080 --no-debug
# cargo run --bin kajiya --release -- --scene pica --width 1920 --height 1080 --no-debug

# --width 2560 --height 1440
# --no-vsync
# --no-window-decorations
