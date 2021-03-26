@echo off
set SMOL_THREADS=64

rem cargo run --bin kajiya --release -- --scene %* --width 1920 --height 1080

cargo run --bin kajiya --release -- --scene pica --width 1920 --height 1080
rem --width 2560 --height 1440
rem --no-vsync
rem --no-decorations
