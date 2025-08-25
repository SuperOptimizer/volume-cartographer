option(VC_BUILD_JSON "Build in-source JSON library" on)
if(VC_BUILD_JSON)
    FetchContent_Declare(
            glaze
            GIT_REPOSITORY https://github.com/stephenberry/glaze.git
            GIT_TAG main
            GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(glaze)
else()
    find_package(glaze REQUIRED)
endif()