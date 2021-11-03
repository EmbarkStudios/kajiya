@echo off

cargo run --features dlss --bin view --release -- --scene %* --no-debug --width 1920 --height 1080
rem cargo run --bin view --release -- --scene pica --width 1920 --height 1080 --no-debug

rem --width 2560 --height 1440
rem --no-vsync
rem --no-window-decorations
