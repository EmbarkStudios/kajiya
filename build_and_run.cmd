@echo off

set SMOL_THREADS=64
rem cargo run --bin vicki --release -- --scene %* --width 1920 --height 1080

cargo run --bin vicki --release -- --scene battle --y-offset=-0.001 --width 1920 --height 1080
rem --width 2560 --height 1440

rem Mitsuba match:
rem cargo run --bin vicki --release -- --scene testball --y-offset=-0.001 --width 1280 --height 720

rem cargo run --bin vicki --release -- --scene "assets/meshes3/draft_punk/scene.gltf" --scale 0.1 --width 1920 --height 1080
rem cargo run --bin vicki --release -- --scene "assets/meshes3/flying_world_-_battle_of_the_trash_god/scene.gltf" --scale 0.005 --width 1920 --height 1080

rem cargo run --bin vicki --release -- --scene "assets/meshes2/rm_342/scene.gltf" --scale 0.03 --width 1920 --height 1080
rem cargo run --bin vicki --release -- --scene "assets/meshes2/336_lrm/scene.gltf" --scale 0.03 --width 1920 --height 1080

rem cargo run --bin vicki --release -- --scene "assets/meshes3/gas_stations_fixed/scene.gltf" --scale 0.01 --width 1920 --height 1080
rem cargo run --bin vicki --release -- --scene "assets/meshes/mireys_cute_gas_stove/scene.gltf" --scale 0.03 --width 1920 --height 1080
rem cargo run --bin vicki --release -- --scene "assets/meshes/sploosh-o-matic/scene.gltf" --scale 0.1

rem cargo run --bin vicki --release -- --scene "assets/meshes/dieselpunk_hovercraft/scene.gltf" --scale 0.01 --width 1920 --height 1080
rem cargo run --bin vicki --release -- --scene "assets/meshes3/viziers_observation_deck/scene.gltf" --scale 0.01 --width 1920 --height 1080
rem cargo run --bin vicki --release -- --scene "assets/meshes3/concept_art_shopping_kid/scene.gltf" --scale 0.1 --width 1920 --height 1080

rem cargo run --bin vicki --release -- --scene "assets/meshes/cornell_box/scene.gltf" --scale 1.0
rem  --width 1920 --height 1080

rem cargo run --bin vicki --release -- --scene "assets/meshes3/sponza/Sponza.gltf" --scale 0.32 --width 1920 --height 1080

rem cargo run --bin vicki --release -- --scene "assets/meshes/pica_pica_-_mini_diorama_01/scene.gltf" --scale 0.1 --width 1920 --height 1080
rem cargo run --bin vicki --release -- --scene "assets/meshes/RpgPackLite/RpgPackLite.gltf" --scale 0.2 --width 1920 --height 1080
rem cargo run --bin vicki --release -- --scene "assets/meshes3/baba_yagas_hut/scene.gltf" --scale 0.333 --width 1920 --height 1080

rem cargo run --bin vicki --release -- --scene "assets/meshes3/tube_city_war/scene.gltf" --scale 1.0 --width 1920 --height 1080

rem cargo run --bin vicki --release -- --scene "assets/test-meshes/gi-01.gltf" --scale 1.0
rem cargo run --bin vicki --release -- --scene "assets/meshes3/walkman/scene.gltf" --scale 10.0 --width 1920 --height 1080
rem cargo run --bin vicki --release -- --scene "assets/meshes3/cozy-room/scene.gltf" --scale 1.0 --width 1920 --height 1080
rem cargo run --bin vicki --release -- --scene "assets/meshes3/kitchen-interior/scene.gltf" --scale 0.75 --width 1920 --height 1080