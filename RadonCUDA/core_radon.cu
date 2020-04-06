#include "pch.h"
#include "framework.h"
#include "RadonCUDA.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>

#define SHARE_BUFF_SIZE 497
#define DEG_TO_RAD  3.14159265358979 / 180.0;
#define MAX(x, y) ((x) > (y) ? (x) : (y))

__device__  inline void incrementRadon(float *pr, const float pixel, const float r)
{
	int r1 = (int)r;
	float delta = r - r1;
	// 此处需要实现原子操作（互斥访问内存），否则计算结果将不正确
	atomicAdd(pr + r1, pixel * (1.0 - delta));
	atomicAdd(pr + r1 + 1, pixel * delta);
}

__global__ void getSine(const float *rad, float *sine, int size)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
		sine[index] = __sinf(rad[index]);
}

__global__ void getCosine(const float *rad, float *cosine, int size)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
		cosine[index] = __cosf(rad[index]);
}

__global__ void getRadVec(const float min, const float inv, float *rad_vec, int size)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
		rad_vec[index] = (min + index * inv) * DEG_TO_RAD;
}

__global__ void getySinTable(const float *sin_vec, const int center_y, float *ySinTable)
{
	int k = threadIdx.x;
	int x = blockIdx.x;
	int y = blockIdx.y;
	float val = center_y - x;
	int offset = k * gridDim.x * gridDim.y + x * gridDim.y + y;  // 注意这里gridDim和blockDim的区别
	if (y == 0) ySinTable[offset] = (val - 0.25) * sin_vec[k];
	else ySinTable[offset] = (val + 0.25) * sin_vec[k];
}

__global__ void getxCosTable(const float *cos_vec, const int center_x, float *xCosTable)
{
	int k = threadIdx.x;
	int x = blockIdx.x;
	int y = blockIdx.y;
	float val = x - center_x;
	int offset = k * gridDim.x * gridDim.y + x * gridDim.y + y;
	if (y == 0) xCosTable[offset] = (val - 0.25) * cos_vec[k];
	else xCosTable[offset] = (val + 0.25) * cos_vec[k];
}

// 此处的len指的是radon矩阵的另一个维度（与角度的数目不同）
__global__ void getRadonMatrix(const float *xCosTable, const float *ySinTable, const int y_row, const int x_col, 
	const float *src_img, const int basic_val, const int len, float *radon_matrix)
{
	__shared__ float temp[4 * SHARE_BUFF_SIZE];
	int i = threadIdx.x;
	int stride = blockDim.x;
	while (i < len)
	{
		temp[4 * i] = 0;
		temp[4 * i + 1] = 0;
		temp[4 * i + 2] = 0;
		temp[4 * i + 3] = 0;
		i += stride;
	}
	__syncthreads();

	int k = blockIdx.x;
	int x = threadIdx.x;
	int y = 0;
	while (y < y_row)
	{
		float pixel = src_img[y * stride + x];
		if (pixel != 0.0)
		{
			pixel *= 0.25;
			float r = 0;

			r = xCosTable[k * 2 * x_col + 2 * x] + ySinTable[k * 2 * y_row + 2 * y] - basic_val;
			incrementRadon(temp, pixel, r);
			r = xCosTable[k * 2 * x_col + 2 * x + 1] + ySinTable[k * 2 * y_row + 2 * y] - basic_val;
			incrementRadon(temp + len, pixel, r);
			r = xCosTable[k * 2 * x_col + 2 * x + 1] + ySinTable[k * 2 * y_row + 2 * y + 1] - basic_val;
			incrementRadon(temp + 3 * len, pixel, r);
			r = xCosTable[k * 2 * x_col + 2 * x] + ySinTable[k * 2 * y_row + 2 * y + 1] - basic_val;
			incrementRadon(temp + 2 * len, pixel, r);
		}
		y++;
	}
	__syncthreads();

	i = threadIdx.x;
	while (i < len)
	{
		radon_matrix[k * len + i] += temp[i];
		radon_matrix[k * len + i] += temp[len + i];
		radon_matrix[k * len + i] += temp[2 * len + i];
		radon_matrix[k * len + i] += temp[3 * len + i];
		i += stride;
	}
}

RADONCUDA_API int _radonCuda(const float agl_min, const float agl_inv, const int size,
	const int y_row, const int x_col, const int center_x, const int center_y, const int basic_val,
	const int len, const float *src_img, float *matrix)
{
	float *dev_rad_vec = nullptr;
	float *dev_sine_vec = nullptr;
	float *dev_sine_tab = nullptr;
	float *dev_cosine_vec = nullptr;
	float *dev_cosine_tab = nullptr;
	float *dev_src_img = nullptr;
	float *dev_radon_matrix = nullptr;

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	/*******************************************************************************/
	/*                                                        计时开始                                                                  */
	/*******************************************************************************/

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	/*******************************************************************************/
	/*                                                          生成弧度向量                                                         */
	/*******************************************************************************/

		// Allocate GPU buffers for dev_rad_vec
	cudaStatus = cudaMalloc((void**)&dev_rad_vec, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	getRadVec << <(size + 63) / 64, 64 >> > (agl_min, agl_inv, dev_rad_vec, size);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "getRadonMatrix launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching getRadonMatrix!\n", cudaStatus);
		goto Error;
	}

	/*******************************************************************************/
	/*                                                          生成Sine矩阵                                                         */
	/*******************************************************************************/

		// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_sine_vec, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	getSine << <(size + 63) / 64, 64 >> > (dev_rad_vec, dev_sine_vec, size);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "getRadonMatrix launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching getRadonMatrix!\n", cudaStatus);
		goto Error;
	}

	/*******************************************************************************/
	/*                                                        生成ySinTable                                                         */
	/*******************************************************************************/

		// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_sine_tab, size * y_row * 2 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	dim3 grid_ysintab(y_row, 2);
	getySinTable << <grid_ysintab, size >> > (dev_sine_vec, center_y, dev_sine_tab);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "getRadonMatrix launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching getRadonMatrix!\n", cudaStatus);
		goto Error;
	}

	/*******************************************************************************/
	/*                                                       生成Cosine矩阵                                                         */
	/*******************************************************************************/

		// Allocate GPU buffers for dev_cosine_vec
	cudaStatus = cudaMalloc((void**)&dev_cosine_vec, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	getCosine << <(size + 63) / 64, 64 >> > (dev_rad_vec, dev_cosine_vec, size);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "getRadonMatrix launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching getRadonMatrix!\n", cudaStatus);
		goto Error;
	}

	/*******************************************************************************/
	/*                                                       生成xCosTable                                                         */
	/*******************************************************************************/

		// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_cosine_tab, size * x_col * 2 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	dim3 grid_xcostab(x_col, 2);
	getxCosTable << <grid_xcostab, size >> > (dev_cosine_vec, center_x, dev_cosine_tab);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "getRadonMatrix launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching getRadonMatrix!\n", cudaStatus);
		goto Error;
	}

	/*******************************************************************************/
	/*                                                       计算radon矩阵                                                          */
	/*******************************************************************************/

		// Allocate GPU buffers for 
	cudaStatus = cudaMalloc((void**)&dev_src_img, y_row * x_col * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate GPU buffers for 
	cudaStatus = cudaMalloc((void**)&dev_radon_matrix, size * len * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_src_img, src_img, y_row * x_col * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

	cudaStatus = cudaMemset(dev_radon_matrix, 0, size * len * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

	getRadonMatrix << <size, x_col >> > (dev_cosine_tab, dev_sine_tab, y_row, x_col, dev_src_img, basic_val, len, dev_radon_matrix);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "getRadonMatrix launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching getRadonMatrix!\n", cudaStatus);
		goto Error;
	}

	/*******************************************************************************/
	/*                                                  将计算结果拷贝回来                                                        */
	/*******************************************************************************/

		// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(matrix, dev_radon_matrix, size * len * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	/*******************************************************************************/
	/*                                                       计时结束                                                                   */
	/*******************************************************************************/
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to generate : %3.lf ms\n", elapsedTime);

Error:
	cudaFree(dev_rad_vec);
	cudaFree(dev_sine_vec);
	cudaFree(dev_sine_tab);
	cudaFree(dev_cosine_vec);
	cudaFree(dev_cosine_tab);
	cudaFree(dev_src_img);
	cudaFree(dev_radon_matrix);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != 0) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}

	return cudaStatus;
}
