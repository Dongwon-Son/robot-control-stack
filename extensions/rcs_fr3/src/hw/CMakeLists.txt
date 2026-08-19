add_library(hw)
target_sources(hw PRIVATE Franka.cpp FrankaHand.cpp FrankaMotionGenerator.cpp TamHook.cpp)
target_link_libraries(
  hw
  PUBLIC franka rcs pinocchio::all Eigen3::Eigen
)

# Standalone TAM hook unit test (Eigen only, no libfranka needed).
option(RCS_FR3_BUILD_TAM_HOOK_TEST "Build the TamHook unit test executable" OFF)
if(RCS_FR3_BUILD_TAM_HOOK_TEST)
  add_executable(tam_hook_test tam_hook_test.cpp TamHook.cpp)
  target_link_libraries(tam_hook_test PRIVATE Eigen3::Eigen)
  target_include_directories(tam_hook_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  enable_testing()
  add_test(NAME tam_hook_test COMMAND tam_hook_test)
endif()

# add_executable(test_hw main.cpp)
# target_link_libraries(test_hw PRIVATE hw)
