// 下列 ifdef 块是创建使从 DLL 导出更简单的
// 宏的标准方法。此 DLL 中的所有文件都是用命令行上定义的 RADONCUDA_EXPORTS
// 符号编译的。在使用此 DLL 的
// 任何项目上不应定义此符号。这样，源文件中包含此文件的任何其他项目都会将
// RADONCUDA_API 函数视为是从 DLL 导入的，而此 DLL 则将用此宏定义的
// 符号视为是被导出的。
#ifdef RADONCUDA_EXPORTS
#define RADONCUDA_API __declspec(dllexport)
#else
#define RADONCUDA_API __declspec(dllimport)
#endif

// agl_min : 角度向量的最小值
// agl_inv : 角度向量的间距
// size : 角度向量元素的数目
// center_y, center_x : 原始图像中心点的坐标
// y_row : 原始图像的行数
// x_col : 原始图像的列数
// basic_val : 计算radon时的基值
// len : radon另一个维度的长度
// matrix : 得到的radon矩阵
extern "C" RADONCUDA_API int _radonCuda(const float agl_min, const float agl_inv, const int size,
	const int y_row, const int x_col, const int center_x, const int center_y, const int basic_val,
	const int len, const float *src_img, float *matrix);
