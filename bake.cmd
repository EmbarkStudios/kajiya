@echo off

set SMOL_THREADS=64

rem cargo run --bin bake --release -- --scene "assets/meshes3/draft_punk/scene.gltf" --scale 0.1 -o sheds
cargo run --bin bake --release -- --scene "assets/meshes3/flying_world_-_battle_of_the_trash_god/scene.gltf" --scale 0.0025 -o battle
rem cargo run --bin bake --release -- --scene "assets/meshes2/rm_342/scene.gltf" --scale 0.015 -o rm_342
rem cargo run --bin bake --release -- --scene "assets/meshes2/336_lrm/scene.gltf" --scale 0.015 -o 336_lrm
rem cargo run --bin bake --release -- --scene "assets/meshes/sploosh-o-matic/scene.gltf" --scale 0.1 -o sploosh-o-matic
rem cargo run --bin bake --release -- --scene "assets/meshes3/sponza/Sponza.gltf" --scale 0.32 -o sponza
rem cargo run --bin bake --release -- --scene "assets/meshes/pica_pica_-_mini_diorama_01/scene.gltf" --scale 0.1 -o pica
