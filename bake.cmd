@echo off

set SMOL_THREADS=64

cargo run --bin bake --release -- --scene "assets/meshes/flying_world_-_battle_of_the_trash_god/scene.gltf" --scale 0.001875 -o battle
cargo run --bin bake --release -- --scene "assets/meshes/336_lrm/scene.gltf" --scale 0.01 -o 336_lrm
cargo run --bin bake --release -- --scene "assets/meshes/pica_pica_-_mini_diorama_01/scene.gltf" --scale 0.1 -o pica
cargo run --bin bake --release -- --scene "assets/meshes/floor/scene.gltf" --scale 1.0 -o floor
cargo run --bin bake --release -- --scene "assets/meshes/testball/scene.gltf" --scale 1.0 -o testball
cargo run --bin bake --release -- --scene "assets/meshes/cornell_box/scene.gltf" --scale 2.0 -o cornell_box
cargo run --bin bake --release -- --scene "assets/meshes/gas_stations_fixed/scene.gltf" --scale 0.005 -o gas_stations
cargo run --bin bake --release -- --scene "assets/meshes/viziers_observation_deck/scene.gltf" --scale 0.0075 -o viziers
cargo run --bin bake --release -- --scene "assets/meshes/dp3_homework_4/scene.gltf" --scale 0.05 -o mini_battle
