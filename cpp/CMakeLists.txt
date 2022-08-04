set(SOURCES
        utils.cpp utils.h RandomEngine.cpp RandomEngine.h
        IndexSet.cpp IndexSet.h KDTree.cpp KDTree.h
        algo/kernels.cpp algo/kernels.h
        algo/cell_kernels.h algo/cell_kernels.cpp
        algo/kernels_gpu.h algo/kernels_gpu.cpp
        algo/AbstractDensityEstimator.h algo/AbstractDensityEstimator.cpp
        algo/VoronoiDensityEstimator.h algo/VoronoiDensityEstimator.cpp
        cl/voronoi.cl cl/voronoi_cl.h)

convertIntoHeader(cl/voronoi.cl cl/voronoi_cl.h voronoi_kernel_sources)

add_library(vde MODULE density_python.cpp ${SOURCES})
target_link_libraries(vde libclew libgpu cnpy ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
target_include_directories(vde PRIVATE ${PYTHON_INCLUDE_DIRS})


add_executable(density_test test/density_test.cpp ${SOURCES})
target_link_libraries(density_test libclew libgpu cnpy)
set_target_properties(density_test PROPERTIES EXCLUDE_FROM_ALL TRUE)
