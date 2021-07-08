cuda_add_library(vectorAdd SHARED
        vectorAdd.cu
        OPTIONS -arch sm_70 -O3 -g
)


add_executable(tutorial8_distribute_dgemm
		tutorial8.cc

		data/data_type.h
		data/matrix_data.h
		data/matrix_block_data.h
		data/cuda_matrix_block_data.h

		state/cuda_input_block_state.h

		cuda_tasks/cuda_copy_in_gpu.h
		cuda_tasks/cuda_copy_out_gpu.h
		cuda_tasks/cuda_product_task.h
		cuda_tasks/cuda_addition_task.h
		cuda_tasks/cuda_addition_task.h
		cuda_tasks/cuda_streams.h

		utils/cuda_utils.h

		${tclap_headers}
		)

target_compile_definitions(tutorial8_distribute_dgemm PUBLIC
		HH_USE_CUDA)

target_link_libraries(tutorial8_distribute_dgemm
		${Hedgehog_LIBRARIES}
		${CUDA_LIBRARIES}
		cuda
		vectorAdd)

target_include_directories(tutorial8_distribute_dgemm PUBLIC
		${Hedgehog_INCLUDE_DIR}
		${CUDA_INCLUDE_DIRS})

CUDA_ADD_CUBLAS_TO_TARGET(tutorial8_distribute_dgemm)

