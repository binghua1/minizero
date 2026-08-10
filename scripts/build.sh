#!/bin/bash
set -e

repo_root=$(readlink -f "$(dirname "$0")/..")
cd "$repo_root"
env_cmakelists="$repo_root/minizero/environment/CMakeLists.txt"
support_games=($(awk '/target_include_directories/,/\)/' ${env_cmakelists} | sed 's|/|\n|g' | grep -v -E 'target|environment|PUBLIC|CMAKE_CURRENT_SOURCE_DIR|base|stochastic|)'))

# Linked worktrees may be mounted into a container without their external
# git-dir. Git metadata is useful for version reporting, but is not required to
# compile MiniZero.
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
	git_hash=$(git log -1 --format=%H)
	git_short_hash=$(git describe --abbrev=6 --dirty --always --exclude '*')
	git config core.hooksPath .githooks || echo "warning: cannot configure Git hooks" >&2
else
	git_hash=xxxxxx
	git_short_hash=xxxxxx
	echo "warning: Git metadata unavailable; building with version xxxxxx" >&2
fi

usage() {
	echo "Usage: $0 GAME_TYPE BUILD_TYPE"
	echo ""
	echo "Required arguments:"
	echo "  GAME_TYPE: $(echo ${support_games[@]} | sed 's/ /, /g')"
	echo "  BUILD_TYPE: release(default), debug"
	exit 1
}

build_game() {
	# check arguments is vaild
	game_type=${1,,}
	build_type=$2
	[[ " ${support_games[*]} " == *" ${game_type} "* ]] || usage
	[ "${build_type}" == "Debug" ] || [ "${build_type}" == "Release" ] || usage

	# check whether the build type and cache are consistent
	if [ -f "build/${game_type}/CMakeCache.txt" ]; then
		cache_build_type=$(grep -oP "CMAKE_BUILD_TYPE:STRING=\K\w+" build/${game_type}/CMakeCache.txt)
		cache_source=$(sed -n 's|^CMAKE_HOME_DIRECTORY:INTERNAL=||p' build/${game_type}/CMakeCache.txt)
		if [ "${cache_build_type}" != "${build_type}" ] || [ "${cache_source}" != "${repo_root}" ]; then
			echo "rebuilding ${game_type}: CMake cache belongs to ${cache_source:-an unknown source path}"
			# Keep generated experiment configs while discarding relocatable build artifacts.
			find "build/${game_type}" -mindepth 1 -maxdepth 1 ! -name '*.cfg' -exec rm -rf -- {} +
		fi
	fi

	# build
	echo "game type: ${game_type}"
	echo "build type: ${build_type}"
	if [ ! -f "build/${game_type}/Makefile" ]; then
		mkdir -p build/${game_type}
		cd build/${game_type}
		cmake ../../ -DCMAKE_BUILD_TYPE=${build_type} -DGAME_TYPE=${game_type^^}
	else
		cd build/${game_type}
	fi

	# create git info file
	mkdir -p git_info
	git_info=$(echo -e "#pragma once\n\n#define GIT_HASH \"${git_hash}\"\n#define GIT_SHORT_HASH \"${git_short_hash}\"")
	if [ ! -f git_info/git_info.h ] || [ $(diff -q <(echo "${git_info}") <(cat git_info/git_info.h) | wc -l 2>/dev/null) -ne 0 ]; then
		echo "${git_info}" > git_info/git_info.h
	fi

	# make
	make -j$(nproc --all)
	cd ../..
}

game_type=${1:-all}
build_type=${2:-release}
build_type=$(echo ${build_type:0:1} | tr '[:lower:]' '[:upper:]')$(echo ${build_type:1} | tr '[:upper:]' '[:lower:]')
[ "${game_type}" == "all" ] && [ ! -d "build" ] && usage

if [ "${game_type}" == "all" ]; then
	for game in build/*
	do
		build_game $(basename ${game}) ${build_type}
	done
else
	build_game ${game_type} ${build_type}
fi
